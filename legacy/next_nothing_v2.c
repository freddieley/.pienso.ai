/*
 * next_nothing_v2.c — Novel Binary-Operations-Only Architectures
 *
 * Four architectures that don't exist yet, tested against the cascade baseline:
 *
 *  CASCADE      (baseline)  PPM cascade: highest-order table with any data wins.
 *
 *  CONF_CASCADE (novel 1)   Same tables, different selection: picks the MOST
 *                           CONFIDENT level (smoothed best/total ratio), not
 *                           the highest-order one. A sparse octagram entry with
 *                           count=1 can't override a dense fourgram with 200
 *                           observations. Solves PPM's cold-start dominance bug.
 *
 *  RESONANCE    (novel 2)   No counting tables at all. Maintains a 512-entry
 *                           ring of (context_hash, next_tok) pairs. Prediction =
 *                           stored tok with minimum Hamming distance to the current
 *                           context hash. Pure XOR + popcount; memory is O(ring).
 *                           Memory usage is ~3 KB vs ~10 MB for cascade.
 *
 *  TEMPORAL     (novel 3)   Three trigram tables, each indexed at a different
 *                           temporal offset in the working ring (offset 0, 2, 6).
 *                           Picks the most confident "lens". Unlike PPM which
 *                           extends one contiguous prefix, TEMPORAL sees three
 *                           separate non-overlapping context windows across time.
 *
 *  ENTANGLED    (novel 4)   One fourgram table whose bucket index is XOR-fused
 *                           with a 32-bit rotating XOR accumulator of ALL past
 *                           tokens. Gives infinite effective context at O(1) cost
 *                           per step—no table size increase, no sliding window.
 *
 *  FUSED        (novel 5)   Merges CONF_CASCADE and ENTANGLED: maintains tri,
 *                           quad, oct, AND an lxor-fused ent table simultaneously.
 *                           conf_score selects among all four levels. When the
 *                           (fourgram ⊕ lxor) context is well-attested, ent wins;
 *                           when sparse, the cascade tables cover. Combines
 *                           short-range richness with infinite-range specificity.
 *
 * All hashing: XOR, shift, OR only. No multiply in the hot prediction path.
 * Training: always external observations only (never own generated output).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================ constants */

#define VOCAB_SIZE   256
#define WORK_RING    1024
#define RES_RING     512
#define CONF_SMOOTH  4U     /* smoothing for confidence scoring */

/* ============================================================ types */

typedef struct { uint8_t token; uint16_t count; } slot_t;
typedef struct { slot_t *slot; } bucket_t;
typedef struct { uint32_t hash; uint8_t tok; uint8_t valid; } res_entry_t;

typedef enum {
    NK_CASCADE      = 0,
    NK_CONF_CASCADE = 1,
    NK_RESONANCE    = 2,
    NK_TEMPORAL     = 3,
    NK_ENTANGLED    = 4,
    NK_FUSED        = 5,
    NK_COUNT        = 6
} nkind_t;

static const char *NK_NAME[NK_COUNT] = {
    "cascade", "conf-cascade", "resonance", "temporal", "entangled", "fused"
};

typedef struct {
    nkind_t  kind;
    uint32_t slots;
    uint32_t ctx_count;     /* must be power of 2 */
    uint32_t mask;          /* ctx_count - 1       */

    uint32_t *unigram;      /* [256]                      */
    uint16_t *bigram;       /* [256 x 256]                */

    /* CASCADE / CONF_CASCADE: three ordered tables */
    bucket_t *tri;
    bucket_t *quad;
    bucket_t *oct;

    /* TEMPORAL: two extra offset tables (offset 0 reuses quad) */
    bucket_t *tbl2;         /* fourgram context at ring-offset 2-5 */
    bucket_t *tbl6;         /* fourgram context at ring-offset 6-9 */

    /* ENTANGLED: one fused-context table + long-range accumulator */
    bucket_t *ent;
    uint32_t  lxor;

    /* RESONANCE: fixed-size observation ring */
    res_entry_t res[RES_RING];
    uint32_t    res_head;

    uint8_t  work[WORK_RING];
    uint32_t work_head;

    uint64_t seen;
    uint64_t correct;
} nrt_t;

/* ============================================================ utilities */

static inline uint16_t sat16(uint16_t v, uint16_t a) {
    uint32_t n = (uint32_t)v + a;
    return n > 65535U ? 65535U : (uint16_t)n;
}
static inline uint8_t recent(const nrt_t *rt, uint32_t back) {
    return rt->work[(rt->work_head + WORK_RING - 1U - back) % WORK_RING];
}
static inline void push(nrt_t *rt, uint8_t tok) {
    rt->work[rt->work_head] = tok;
    rt->work_head = (rt->work_head + 1U) % WORK_RING;
}

/* Pure bit-operations popcount — no multiply, no library call. */
static inline uint32_t pop32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    x += x >> 8;
    x += x >> 16;
    return x & 63U;
}

/* ============================================================ XOR-shift hashes
   All context hashing uses only XOR (^) and bit-shift (>>, <<).
   No multiplication in the prediction hot path.                              */

static inline uint16_t h_tri(uint8_t a, uint8_t b, uint32_t mask) {
    uint32_t h = ((uint32_t)a << 8) | (uint32_t)b;
    h ^= h << 7; h ^= h >> 3; h ^= h << 5;
    return (uint16_t)(h & mask);
}
static inline uint16_t h_quad(uint8_t a, uint8_t b, uint8_t c, uint32_t mask) {
    uint32_t h = ((uint32_t)a << 16) | ((uint32_t)b << 8) | (uint32_t)c;
    h ^= h << 5; h ^= h >> 7; h ^= h << 9;
    return (uint16_t)(h & mask);
}
static inline uint32_t h_oct7(uint8_t t7, uint8_t t6, uint8_t t5, uint8_t t4,
                               uint8_t t3, uint8_t t2, uint8_t t1) {
    uint32_t h = ((uint32_t)t7 << 24) | ((uint32_t)t6 << 16)
               | ((uint32_t)t5 << 8)  |  (uint32_t)t4;
    h ^= ((uint32_t)t3 << 16) | ((uint32_t)t2 << 8) | (uint32_t)t1;
    h ^= h << 5; h ^= h >> 11; h ^= h << 13;
    return h;
}
/* Entangled: short fourgram context XOR-fused with long-range accumulator. */
static inline uint16_t h_ent(uint8_t a, uint8_t b, uint8_t c,
                              uint32_t lxor, uint32_t mask) {
    uint32_t s = h_quad(a, b, c, 0xFFFFU);
    uint32_t l = lxor ^ (lxor >> 16);
    return (uint16_t)((s ^ l) & mask);
}
/* Long-XOR accumulator update: ROTL32(1) then XOR with token — binary only. */
static inline uint32_t lxor_upd(uint32_t acc, uint8_t tok) {
    return ((acc << 1) | (acc >> 31)) ^ (uint32_t)tok;
}

/* ============================================================ bucket ops */

static void b_inc(bucket_t *b, uint32_t slots, uint8_t tok) {
    int fi = -1, mi = 0;
    uint16_t mc = b->slot[0].count;
    for (uint32_t i = 0; i < slots; ++i) {
        if (!b->slot[i].count && fi < 0) fi = (int)i;
        if (b->slot[i].count && b->slot[i].token == tok) {
            b->slot[i].count = sat16(b->slot[i].count, 1U);
            return;
        }
        if (b->slot[i].count < mc) { mc = b->slot[i].count; mi = (int)i; }
    }
    if (fi >= 0) { b->slot[fi].token = tok; b->slot[fi].count = 1U; return; }
    if (b->slot[mi].count > 1U) b->slot[mi].count--;
    else { b->slot[mi].token = tok; b->slot[mi].count = 1U; }
}

/* Fills tok/total/best. Returns 0 if bucket empty. */
static int b_pred(const bucket_t *b, uint32_t slots,
                  uint8_t *tok_out, uint32_t *tot, uint32_t *best) {
    uint32_t t = 0, bst = 0;
    uint8_t btok = 32U;
    for (uint32_t i = 0; i < slots; ++i) {
        t += b->slot[i].count;
        if (b->slot[i].count > bst) { bst = b->slot[i].count; btok = b->slot[i].token; }
    }
    if (!t) return 0;
    *tok_out = btok; *tot = t; *best = bst;
    return 1;
}

/* Smoothed confidence: high count AND high ratio both matter.
   Prevents cold-start entries from winning on perfect-but-tiny evidence. */
static inline uint32_t conf_score(uint32_t best, uint32_t total) {
    return (best << 16) / (total + CONF_SMOOTH);
}

/* ============================================================ allocation */

static bucket_t *alloc_tbl(uint32_t n, uint32_t slots) {
    bucket_t *t = (bucket_t *)calloc(n, sizeof(bucket_t));
    if (!t) return NULL;
    for (uint32_t i = 0; i < n; ++i) {
        t[i].slot = (slot_t *)calloc(slots, sizeof(slot_t));
        if (!t[i].slot) return NULL;
    }
    return t;
}
static void free_tbl(bucket_t *t, uint32_t n) {
    if (!t) return;
    for (uint32_t i = 0; i < n; ++i) free(t[i].slot);
    free(t);
}

static int nrt_init(nrt_t *rt, nkind_t kind, uint32_t slots, uint32_t ctx_count) {
    memset(rt, 0, sizeof(*rt));
    if (!ctx_count || (ctx_count & (ctx_count - 1))) return 0; /* must be pow2 */
    rt->kind = kind; rt->slots = slots;
    rt->ctx_count = ctx_count; rt->mask = ctx_count - 1U;

    rt->unigram = (uint32_t *)calloc(VOCAB_SIZE, sizeof(uint32_t));
    rt->bigram  = (uint16_t *)calloc((size_t)VOCAB_SIZE * VOCAB_SIZE, sizeof(uint16_t));
    if (!rt->unigram || !rt->bigram) return 0;

    switch (kind) {
        case NK_CASCADE: case NK_CONF_CASCADE:
            rt->tri  = alloc_tbl(ctx_count, slots);
            rt->quad = alloc_tbl(ctx_count, slots);
            rt->oct  = alloc_tbl(ctx_count, slots);
            if (!rt->tri || !rt->quad || !rt->oct) return 0;
            break;
        case NK_TEMPORAL:
            rt->tri  = alloc_tbl(ctx_count, slots);
            rt->tbl2 = alloc_tbl(ctx_count, slots);
            rt->tbl6 = alloc_tbl(ctx_count, slots);
            if (!rt->tri || !rt->tbl2 || !rt->tbl6) return 0;
            break;
        case NK_ENTANGLED:
            rt->ent = alloc_tbl(ctx_count, slots);
            if (!rt->ent) return 0;
            break;
        case NK_FUSED:
            rt->tri  = alloc_tbl(ctx_count, slots);
            rt->quad = alloc_tbl(ctx_count, slots);
            rt->oct  = alloc_tbl(ctx_count, slots);
            rt->ent  = alloc_tbl(ctx_count, slots);
            if (!rt->tri || !rt->quad || !rt->oct || !rt->ent) return 0;
            break;
        case NK_RESONANCE:
            break; /* ring is inline in struct; no heap tables */
        default: return 0;
    }
    return 1;
}

static void nrt_free(nrt_t *rt) {
    free(rt->unigram); free(rt->bigram);
    free_tbl(rt->tri,  rt->ctx_count);
    free_tbl(rt->quad, rt->ctx_count);
    free_tbl(rt->oct,  rt->ctx_count);
    free_tbl(rt->tbl2, rt->ctx_count);
    free_tbl(rt->tbl6, rt->ctx_count);
    free_tbl(rt->ent,  rt->ctx_count);
    memset(rt, 0, sizeof(*rt));
}

/* ============================================================ bigram fallback */

#define BG(rt,a,b) ((rt)->bigram[(uint32_t)(a)*VOCAB_SIZE+(uint32_t)(b)])

static uint8_t bigram_pred(const nrt_t *rt, uint8_t t1, uint8_t *conf_out) {
    uint32_t tot = 0, bst = 0;
    uint8_t btok = 32U;
    for (uint32_t i = 0; i < VOCAB_SIZE; ++i) {
        uint16_t c = BG(rt, t1, i);
        tot += c;
        if (c > bst) { bst = c; btok = (uint8_t)i; }
    }
    *conf_out = tot ? (uint8_t)((bst * 255U) / tot) : 0U;
    return btok;
}

/* ============================================================ update functions */

static void upd_cascade(nrt_t *rt, uint8_t obs) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    uint32_t m = rt->mask;
    rt->unigram[obs]++;
    BG(rt, t1, obs) = sat16(BG(rt, t1, obs), 1U);
    b_inc(&rt->tri [h_tri (t2,t1,m)],                     rt->slots, obs);
    b_inc(&rt->quad[h_quad(t3,t2,t1,m)],                  rt->slots, obs);
    b_inc(&rt->oct [h_oct7(t7,t6,t5,t4,t3,t2,t1) & m],   rt->slots, obs);
}

static void upd_temporal(nrt_t *rt, uint8_t obs) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5);
    uint8_t t7=recent(rt,6), t8=recent(rt,7), t9=recent(rt,8);
    uint32_t m = rt->mask;
    rt->unigram[obs]++;
    BG(rt, t1, obs) = sat16(BG(rt, t1, obs), 1U);
    b_inc(&rt->tri [h_quad(t3,t2,t1,m)],    rt->slots, obs);  /* offset 0 (immediate) */
    b_inc(&rt->tbl2[h_quad(t6,t5,t4,m)],    rt->slots, obs);  /* offset 3 (mid)       */
    b_inc(&rt->tbl6[h_quad(t9,t8,t7,m)],    rt->slots, obs);  /* offset 6 (far)       */
}

static void upd_entangled(nrt_t *rt, uint8_t obs) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint32_t m = rt->mask;
    rt->unigram[obs]++;
    BG(rt, t1, obs) = sat16(BG(rt, t1, obs), 1U);
    b_inc(&rt->ent[h_ent(t3,t2,t1,rt->lxor,m)], rt->slots, obs);
    rt->lxor = lxor_upd(rt->lxor, obs);
}

static void upd_fused(nrt_t *rt, uint8_t obs) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    uint32_t m = rt->mask;
    rt->unigram[obs]++;
    BG(rt, t1, obs) = sat16(BG(rt, t1, obs), 1U);
    b_inc(&rt->tri [h_tri (t2,t1,m)],                    rt->slots, obs);
    b_inc(&rt->quad[h_quad(t3,t2,t1,m)],                 rt->slots, obs);
    b_inc(&rt->oct [h_oct7(t7,t6,t5,t4,t3,t2,t1) & m],  rt->slots, obs);
    b_inc(&rt->ent [h_ent (t3,t2,t1,rt->lxor,m)],        rt->slots, obs);
    rt->lxor = lxor_upd(rt->lxor, obs);
}

static void upd_resonance(nrt_t *rt, uint8_t obs) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    rt->unigram[obs]++;
    BG(rt, t1, obs) = sat16(BG(rt, t1, obs), 1U);
    uint32_t h = h_oct7(t7,t6,t5,t4,t3,t2,t1);
    rt->res[rt->res_head] = (res_entry_t){ .hash=h, .tok=obs, .valid=1 };
    rt->res_head = (rt->res_head + 1U) % RES_RING;
}

/* ============================================================ predict functions */

static uint8_t pred_cascade(const nrt_t *rt, uint8_t *co) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    uint32_t m = rt->mask;
    uint8_t tok; uint32_t tot, bst;
    if (b_pred(&rt->oct [h_oct7(t7,t6,t5,t4,t3,t2,t1)&m], rt->slots,&tok,&tot,&bst))
        { *co=(uint8_t)(bst*255U/tot); return tok; }
    if (b_pred(&rt->quad[h_quad(t3,t2,t1,m)],              rt->slots,&tok,&tot,&bst))
        { *co=(uint8_t)(bst*255U/tot); return tok; }
    if (b_pred(&rt->tri [h_tri (t2,t1,m)],                 rt->slots,&tok,&tot,&bst))
        { *co=(uint8_t)(bst*255U/tot); return tok; }
    return bigram_pred(rt, t1, co);
}

/* Threshold for «good enough»: fourgram that's been seen 16+ times with 70%
   majority is more reliable than a fresh octagram hit — stop searching there.  */
#define CONF_ENOUGH 0xB000U

static uint8_t pred_conf_cascade(const nrt_t *rt, uint8_t *co) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    uint32_t m = rt->mask;
    uint8_t tok; uint32_t tot, bst;
    uint32_t best_sc = 0;
    uint8_t  best_tok = 32U, best_co = 0;

    /* Evaluate levels from highest to lowest order; stop early when confident. */
    if (b_pred(&rt->oct [h_oct7(t7,t6,t5,t4,t3,t2,t1)&m], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc >= CONF_ENOUGH) { *co = best_co; return best_tok; }
    if (b_pred(&rt->quad[h_quad(t3,t2,t1,m)], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc >= CONF_ENOUGH) { *co = best_co; return best_tok; }
    if (b_pred(&rt->tri [h_tri(t2,t1,m)],     rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc) { *co = best_co; return best_tok; }
    return bigram_pred(rt, t1, co);
}

static uint8_t pred_temporal(const nrt_t *rt, uint8_t *co) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5);
    uint8_t t7=recent(rt,6), t8=recent(rt,7), t9=recent(rt,8);
    uint32_t m = rt->mask;
    uint8_t tok; uint32_t tot, bst;
    uint32_t best_sc = 0;
    uint8_t  best_tok = 32U, best_co = 0;

    if (b_pred(&rt->tri [h_quad(t3,t2,t1,m)], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (b_pred(&rt->tbl2[h_quad(t6,t5,t4,m)], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (b_pred(&rt->tbl6[h_quad(t9,t8,t7,m)], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc) { *co = best_co; return best_tok; }
    return bigram_pred(rt, t1, co);
}

static uint8_t pred_entangled(const nrt_t *rt, uint8_t *co) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint32_t m = rt->mask;
    uint8_t tok; uint32_t tot, bst;
    if (b_pred(&rt->ent[h_ent(t3,t2,t1,rt->lxor,m)], rt->slots,&tok,&tot,&bst)) {
        *co = (uint8_t)(bst*255U/tot); return tok;
    }
    return bigram_pred(rt, t1, co);
}

static uint8_t pred_fused(const nrt_t *rt, uint8_t *co) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    uint32_t m = rt->mask;
    uint8_t tok; uint32_t tot, bst;
    uint32_t best_sc = 0;
    uint8_t  best_tok = 32U, best_co = 0;

    /* octagram first — if very confident, skip remaining tables */
    if (b_pred(&rt->oct [h_oct7(t7,t6,t5,t4,t3,t2,t1)&m], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc >= CONF_ENOUGH) { *co = best_co; return best_tok; }

    /* lxor-fused ent competes with fourgram on equal footing — checked second */
    if (b_pred(&rt->ent [h_ent(t3,t2,t1,rt->lxor,m)], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc >= CONF_ENOUGH) { *co = best_co; return best_tok; }

    if (b_pred(&rt->quad[h_quad(t3,t2,t1,m)], rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc >= CONF_ENOUGH) { *co = best_co; return best_tok; }

    if (b_pred(&rt->tri [h_tri(t2,t1,m)],     rt->slots,&tok,&tot,&bst)) {
        uint32_t sc = conf_score(bst, tot);
        if (sc > best_sc) { best_sc=sc; best_tok=tok; best_co=(uint8_t)(bst*255U/tot); }
    }
    if (best_sc) { *co = best_co; return best_tok; }
    return bigram_pred(rt, t1, co);
}

static uint8_t pred_resonance(const nrt_t *rt, uint8_t *co) {
    uint8_t t1=recent(rt,0), t2=recent(rt,1), t3=recent(rt,2);
    uint8_t t4=recent(rt,3), t5=recent(rt,4), t6=recent(rt,5), t7=recent(rt,6);
    uint32_t query = h_oct7(t7,t6,t5,t4,t3,t2,t1);
    uint32_t best_dist = 33U;
    uint8_t  best_tok  = 32U;

    for (uint32_t i = 0; i < RES_RING; ++i) {
        if (!rt->res[i].valid) continue;
        uint32_t d = pop32(query ^ rt->res[i].hash);
        if (d < best_dist) { best_dist = d; best_tok = rt->res[i].tok; }
    }
    if (best_dist <= 32U) {
        *co = (best_dist == 0) ? 255U : (uint8_t)(255U >> (best_dist >> 1));
        return best_tok;
    }
    return bigram_pred(rt, t1, co);
}

/* ============================================================ unified interface */

static void nrt_update(nrt_t *rt, uint8_t obs) {
    switch (rt->kind) {
        case NK_CASCADE: case NK_CONF_CASCADE: upd_cascade(rt, obs);   break;
        case NK_TEMPORAL:                      upd_temporal(rt, obs);  break;
        case NK_ENTANGLED:                     upd_entangled(rt, obs); break;
        case NK_FUSED:                         upd_fused(rt, obs);     break;
        case NK_RESONANCE:                     upd_resonance(rt, obs); break;
        default: break;
    }
    push(rt, obs);
}

static uint8_t nrt_predict(const nrt_t *rt, uint8_t *co) {
    switch (rt->kind) {
        case NK_CASCADE:      return pred_cascade(rt, co);
        case NK_CONF_CASCADE: return pred_conf_cascade(rt, co);
        case NK_TEMPORAL:     return pred_temporal(rt, co);
        case NK_ENTANGLED:    return pred_entangled(rt, co);
        case NK_FUSED:        return pred_fused(rt, co);
        case NK_RESONANCE:    return pred_resonance(rt, co);
        default: { *co = 0; return 32U; }
    }
}

/* Push a generated token into context without updating learning tables.
   Also advances lxor for entangled/fused so generation stays coherent. */
static void nrt_push_ctx(nrt_t *rt, uint8_t tok) {
    push(rt, tok);
    if (rt->kind == NK_ENTANGLED || rt->kind == NK_FUSED)
        rt->lxor = lxor_upd(rt->lxor, tok);
}

/* ============================================================ benchmark */

typedef struct {
    char     name[20];
    double   in_acc;
    double   out_acc;
    double   avg_acc;
    double   tps;
    double   mem_mb;
    double   score;
} bresult_t;

static double model_mem_mb(nkind_t kind, uint32_t ctx_count, uint32_t slots) {
    double base  = VOCAB_SIZE * sizeof(uint32_t)
                 + (double)VOCAB_SIZE * VOCAB_SIZE * sizeof(uint16_t);
    double tbl   = (double)ctx_count * slots * sizeof(slot_t);
    double ring  = RES_RING * sizeof(res_entry_t);
    switch (kind) {
        case NK_CASCADE: case NK_CONF_CASCADE: return (base + 3.0*tbl) / (1<<20);
        case NK_TEMPORAL:                      return (base + 3.0*tbl) / (1<<20);
        case NK_FUSED:                         return (base + 4.0*tbl) / (1<<20);
        case NK_ENTANGLED:                     return (base + 1.0*tbl) / (1<<20);
        case NK_RESONANCE:                     return (base + ring)    / (1<<20);
        default: return 0;
    }
}

static double run_stream(nrt_t *rt, const unsigned char *s, uint32_t len,
                         uint32_t warmup, double min_secs, double *tps_out) {
    if (!len) { *tps_out = 0; return 0; }
    /* warmup phase — update but don't measure */
    for (uint32_t i = 0; i < warmup; ++i) nrt_update(rt, s[i % len]);

    uint64_t steps = 0, correct = 0;
    clock_t t0 = clock(), t1;
    do {
        uint32_t n = len > 8000U ? len : 8000U;
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t co = 0;
            uint8_t pred = nrt_predict(rt, &co);
            uint8_t obs  = s[steps % len];
            correct += (pred == obs);
            steps++;
            nrt_update(rt, obs);
        }
        t1 = clock();
    } while (((double)(t1 - t0) / CLOCKS_PER_SEC) < min_secs);

    *tps_out = (double)steps / ((double)(t1 - t0) / CLOCKS_PER_SEC);
    return (double)correct / (double)steps;
}

static bresult_t benchmark(nkind_t kind, uint32_t slots, uint32_t ctx_count,
                           uint32_t boot_passes,
                           const unsigned char *train, uint32_t train_len,
                           const unsigned char *gen,   uint32_t gen_len) {
    bresult_t r;
    memset(&r, 0, sizeof(r));
    snprintf(r.name, sizeof(r.name), "%s", NK_NAME[kind]);

    nrt_t rt;
    if (!nrt_init(&rt, kind, slots, ctx_count)) {
        fprintf(stderr, "init failed: %s\n", r.name);
        return r;
    }
    for (uint32_t p = 0; p < boot_passes; ++p)
        for (uint32_t i = 0; i < train_len; ++i) nrt_update(&rt, train[i]);

    double tps_in, tps_out;
    r.in_acc  = run_stream(&rt, train, train_len, 0U, 0.25, &tps_in);
    r.out_acc = run_stream(&rt, gen,   gen_len,   0U, 0.25, &tps_out);
    r.avg_acc = 0.5 * (r.in_acc + r.out_acc);
    r.tps     = 0.5 * (tps_in + tps_out);
    r.mem_mb  = model_mem_mb(kind, ctx_count, slots);

    nrt_free(&rt);
    return r;
}

static void print_results(bresult_t *arr, uint32_t n) {
    double max_tps = 0, min_mem = 1e18;
    for (uint32_t i = 0; i < n; ++i) {
        if (arr[i].tps    > max_tps) max_tps = arr[i].tps;
        if (arr[i].mem_mb < min_mem) min_mem = arr[i].mem_mb;
    }
    if (max_tps <= 0) max_tps = 1;
    if (min_mem <= 0) min_mem = 1e-6;

    double base_acc = arr[0].avg_acc; /* cascade is index 0 */
    for (uint32_t i = 0; i < n; ++i) {
        double sp = arr[i].tps / max_tps;
        double me = min_mem / arr[i].mem_mb; if (me > 1.0) me = 1.0;
        arr[i].score = 0.65 * arr[i].avg_acc + 0.25 * sp + 0.10 * me;
    }

    printf("\n%-16s  in%%    out%%   avg%%   Mtok/s  mem(MB)  score   delta\n",
           "architecture");
    printf("----------------  -----  -----  -----  ------  -------  ------  -----\n");
    for (uint32_t i = 0; i < n; ++i) {
        double d = arr[i].avg_acc - base_acc;
        printf("%-16s  %5.1f  %5.1f  %5.1f  %6.1f  %7.3f  %0.4f  %+.1f%%\n",
               arr[i].name,
               arr[i].in_acc  * 100.0,
               arr[i].out_acc * 100.0,
               arr[i].avg_acc * 100.0,
               arr[i].tps / 1e6,
               arr[i].mem_mb,
               arr[i].score,
               d * 100.0);
    }
}

/* ============================================================ corpus */

static const unsigned char *embedded_conv(void) {
    return (const unsigned char *)
        "USER: What are we building?\n"
        "ASSISTANT: A simple always-on intelligence core that keeps learning safely.\n"
        "USER: What is the limiting factor?\n"
        "ASSISTANT: Energy per useful bit learned, not raw parameter count.\n"
        "USER: How do we avoid collapse?\n"
        "ASSISTANT: Never train on self-generated text and isolate noisy exploration.\n"
        "USER: What matters most?\n"
        "ASSISTANT: Stability first, then adaptation speed, then scaling.\n"
        "USER: Should it generalize?\n"
        "ASSISTANT: Yes, by preserving simple rules and reusable memory interfaces.\n"
        "USER: How do we make it cheap?\n"
        "ASSISTANT: Binary operations only, online updates, explicit memory with fixed bounds.\n"
        "USER: What is the goal?\n"
        "ASSISTANT: Build a minimal engine that compounds capability without compounding cost.\n";
}

static const unsigned char *embedded_gen(void) {
    return (const unsigned char *)
        "the system learns from direct experience over continuous time always. "
        "stable online learning demands clean external feedback loops only here. "
        "binary operations can build rich behavior with compact fixed state. "
        "prediction quality improves when context is specific and completely fresh. "
        "resource limits force elegant design and explicit tradeoffs every time. "
        "patterns emerge from repetition and frequency across the entire stream. "
        "context windows of different lengths capture different scales of structure. ";
}

static unsigned char *load_file(const char *path, uint32_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f); buf[sz] = '\0'; *len_out = (uint32_t)sz;
    return buf;
}

/* ============================================================ chat showcase */

static void prime_model(nrt_t *rt, const unsigned char *text, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) nrt_update(rt, text[i]);
}

/* Push tokens into context ring WITHOUT updating any counting tables.
   Use this during inference to give context from the prompt,
   so the model can predict the next token without learning from the prompt. */
static void prime_ctx(nrt_t *rt, const unsigned char *text, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) nrt_push_ctx(rt, text[i]);
}

static void gen_reply(nrt_t *rt, const char *prompt, char *out, uint32_t cap) {
    uint32_t len = 0;
    uint8_t prev = 0, run = 0, co;

    /* Provide prompt as context only — never train on it. */
    prime_ctx(rt, (const unsigned char *)"USER: ", 6);
    prime_ctx(rt, (const unsigned char *)prompt, (uint32_t)strlen(prompt));
    prime_ctx(rt, (const unsigned char *)"\nASSISTANT: ", 13);

    while (len + 1U < cap) {
        uint8_t tok = nrt_predict(rt, &co);
        if (!tok) tok = ' ';

        /* Single-char run detection */
        if (tok == prev) run++; else run = 0;
        prev = tok;
        if (run > 6U)  tok = ' ';
        if (run > 12U && len > 16U) break;

        /* Phrase-loop detection: check if last W chars appear anywhere earlier.
           Catches repeating phrases of ANY period up to W chars.              */
        if (len >= 32U) {
            uint32_t wlen = len < 64U ? 12U : 16U;
            int looped = 0;
            for (uint32_t j = 0; j + wlen <= len - wlen; ++j) {
                if (memcmp(out + len - wlen, out + j, wlen) == 0) { looped = 1; break; }
            }
            if (looped) break;
        }

        out[len++] = (char)tok;
        nrt_push_ctx(rt, tok);   /* context advances; tables do not learn */

        if (tok == '\n' && len > 28U) break;
        if (len > 200U) break;
        if ((tok == '.' || tok == '!' || tok == '?') && len > 40U) break;
    }
    out[len] = '\0';
}

static void chat_test(const unsigned char *train, uint32_t train_len,
                      uint32_t slots, uint32_t ctx_count, uint32_t boot_passes,
                      const char *prompt) {
    printf("\n=== \"%s\" ===\n", prompt);
    for (uint32_t k = 0; k < NK_COUNT; ++k) {
        nrt_t rt;
        if (!nrt_init(&rt, (nkind_t)k, slots, ctx_count)) { printf("%-14s: init failed\n", NK_NAME[k]); continue; }
        for (uint32_t p = 0; p < boot_passes; ++p) prime_model(&rt, train, train_len);
        char out[256];
        gen_reply(&rt, prompt, out, sizeof(out));
        printf("%-14s: %s\n", NK_NAME[k], out);
        nrt_free(&rt);
    }
}

/* ============================================================ main */

int main(void) {
    uint32_t ext_len = 0;
    unsigned char *ext = load_file("conversation_train.txt", &ext_len);

    const unsigned char *train = ext ? ext : embedded_conv();
    uint32_t train_len = ext ? ext_len : (uint32_t)strlen((const char *)train);
    const unsigned char *gen  = embedded_gen();
    uint32_t gen_len = (uint32_t)strlen((const char *)gen);

    const uint32_t SLOTS     = 16U;
    const uint32_t CTX_COUNT = 32768U;  /* power of 2 */
    const uint32_t BOOT      = 8U;

    printf("next_nothing_v2: novel binary-only architecture sweep\n");
    printf("train: %s (%u B)  gen: %u B  slots=%u ctx=%u boot=%u\n\n",
           ext ? "conversation_train.txt" : "embedded", train_len, gen_len,
           SLOTS, CTX_COUNT, BOOT);

    bresult_t results[NK_COUNT];
    for (uint32_t k = 0; k < NK_COUNT; ++k) {
        printf("  benchmarking %-14s ...\r", NK_NAME[k]);
        fflush(stdout);
        results[k] = benchmark((nkind_t)k, SLOTS, CTX_COUNT, BOOT,
                               train, train_len, gen, gen_len);
    }

    print_results(results, NK_COUNT);

    /* Chat showcase: same three prompts, all five models answer side-by-side */
    chat_test(train, train_len, SLOTS, CTX_COUNT, BOOT,
              "What is the limiting factor for this architecture?");
    chat_test(train, train_len, SLOTS, CTX_COUNT, BOOT,
              "How do we scale without losing stability?");
    chat_test(train, train_len, SLOTS, CTX_COUNT, BOOT,
              "What should we build next?");

    if (ext) free(ext);
    return 0;
}
