/*
 * word_lm.c — Word-level n-gram language model, pure integer/bitwise/XOR
 *
 * Words are mapped to uint16_t IDs via FNV-1a hash.
 * Hash tables over word-ID sequences replace byte n-gram tables.
 * Result: models learn "neural networks learn" → "by" directly,
 * instead of byte-by-byte which needs many more passes.
 *
 * Architecture:
 *   - 16-bit word IDs (65536 vocab buckets)
 *   - Bigram / Trigram / Quadgram hash tables
 *   - Long-range XOR accumulator (lxor) for infinite effective context
 *   - Same confidence-cascade + early-exit as full_lm.c
 *   - No floats anywhere
 *
 * Model file format (magic 0x574C4D32 "WLM2"):
 *   Header: magic(4) version(4) vocab_size(4) ctx_slots(4) slots(4)
 *   vocab: vocab_size * WORD_MAXLEN (char) — word strings (null-padded)
 *   unigram: vocab_size * uint32_t
 *   bi_tok / bi_cnt / tri_tok / tri_cnt / quad_tok / quad_cnt: ctx_slots*slots each
 *
 * Commands:
 *   word_lm train      <corpus> <model> [passes=4]
 *   word_lm resume     <model>  <corpus> [passes=4]
 *   word_lm train-list <list>   <model> [passes=4]
 *   word_lm eval       <model>  <corpus>
 *   word_lm gen        <model>  <prompt> [max_words=64]
 *   word_lm gen-sample <model>  <prompt> [max_words=64] [temp=1] [seed=0]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── constants ──────────────────────────────────────────────────────────── */
#define WORD_MAGIC     0x574C4D32U  /* "WLM2" */
#define WORD_VERSION   1U
#define VOCAB_SIZE     65536U       /* 16-bit word IDs */
#define WORD_MAXLEN    32U          /* max chars stored per vocab entry */
#define DEFAULT_SLOTS  16U          /* candidates per hash bucket */
#define DEFAULT_CTX    65536U       /* number of context hash buckets */
#define CONF_SMOOTH    4U
#define CONF_ENOUGH    0xB000U      /* early-exit threshold */
#define CHUNK_SZ       (1<<20)      /* 1 MB streaming read */

/* ── types ──────────────────────────────────────────────────────────────── */
typedef struct {
    /* vocabulary: ID → string (heap) */
    char    (*vocab)[WORD_MAXLEN]; /* VOCAB_SIZE × WORD_MAXLEN, heap */

    /* frequency table (heap) */
    uint32_t *unigram;             /* VOCAB_SIZE uint32_t, heap */

    /* hash tables: ctx_slots × slots entries (heap) */
    uint16_t *bi_tok,  *bi_cnt;
    uint16_t *tri_tok, *tri_cnt;
    uint16_t *quad_tok,*quad_cnt;

    /* XOR long-range accumulator */
    uint32_t lxor;

    /* context ring buffer */
    uint16_t ctx_ring[16];
    uint32_t ctx_head;

    uint32_t slots;
    uint32_t ctx_slots;
    uint32_t mask;
} word_lm_t;

/* ── tokeniser ──────────────────────────────────────────────────────────── */
/* returns 1 if c is a valid word character (lowercase in-place) */
static int word_char(int c, char *lc) {
    if (c >= 'A' && c <= 'Z') { *lc = (char)(c + 32); return 1; }
    if (c >= 'a' && c <= 'z') { *lc = (char)c; return 1; }
    if (c >= '0' && c <= '9') { *lc = (char)c; return 1; }
    if (c == '\'')             { *lc = '\'';    return 1; }
    return 0;
}

/* FNV-1a: computes initial probe position (not final ID) */
static uint16_t word_hash(const char *w, uint32_t len) {
    uint32_t h = 2166136261U;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)w[i];
        h += (h << 1) ^ (h << 4) ^ (h << 7) ^ (h << 8) ^ (h << 24);
    }
    return (uint16_t)(h & 0xFFFFU);
}

/* Open-addressing vocab lookup with linear probing.
 * Returns slot index as the word ID (guaranteed unique per distinct word).
 * If insert=1 and word is new, inserts it and returns slot.
 * Returns 0xFFFF on table full or not-found (insert=0). */
static uint16_t vocab_id(char (*vocab)[WORD_MAXLEN], const char *w, uint32_t len, int insert) {
    uint16_t start = word_hash(w, len);
    uint16_t slot  = start;
    do {
        if (vocab[slot][0] == '\0') {
            /* empty slot */
            if (!insert) return 0xFFFFU;
            uint32_t slen = len < (WORD_MAXLEN - 1U) ? len : (WORD_MAXLEN - 1U);
            memcpy(vocab[slot], w, slen);
            vocab[slot][slen] = '\0';
            return slot;
        }
        /* compare exact string */
        if (strncmp(vocab[slot], w, WORD_MAXLEN - 1U) == 0 && (uint32_t)strlen(vocab[slot]) == len)
            return slot;
        slot = (uint16_t)((slot + 1U) & 0xFFFFU); /* linear probe */
    } while (slot != start);
    return 0xFFFFU; /* table full */
}

/* ── context ring ───────────────────────────────────────────────────────── */
static void push_word(word_lm_t *lm, uint16_t wid) {
    lm->ctx_ring[lm->ctx_head & 15U] = wid;
    lm->ctx_head++;
}
static uint16_t recent_word(const word_lm_t *lm, uint32_t ago) {
    return lm->ctx_ring[(lm->ctx_head - 1U - ago) & 15U];
}

/* ── XOR long-range accumulator ─────────────────────────────────────────── */
static uint32_t lxor_upd(uint32_t lx, uint16_t wid) {
    uint32_t v = (uint32_t)wid ^ (lx >> 3) ^ (lx << 5);
    return v ^ (v >> 7);
}

/* ── hash functions (XOR/shift only) ───────────────────────────────────── */
static uint32_t h_bi(uint16_t t1, uint32_t mask) {
    uint32_t h = (uint32_t)t1;
    h ^= h << 7; h ^= h >> 9; h ^= h << 3;
    return h & mask;
}
static uint32_t h_tri(uint16_t t2, uint16_t t1, uint32_t mask) {
    uint32_t h = ((uint32_t)t2 << 16) ^ (uint32_t)t1;
    h ^= h >> 11; h ^= h << 7; h ^= h >> 5;
    return h & mask;
}
static uint32_t h_quad(uint16_t t3, uint16_t t2, uint16_t t1, uint32_t mask) {
    uint32_t h = ((uint32_t)t3 << 16) ^ ((uint32_t)t2 << 8) ^ (uint32_t)t1;
    h ^= h >> 13; h ^= h << 5; h ^= h >> 9;
    return h & mask;
}
/* lxor-fused long-range context */
static uint32_t h_ent(uint16_t t2, uint16_t t1, uint32_t lx, uint32_t mask) {
    uint32_t h = ((uint32_t)t2 << 16) ^ (uint32_t)t1 ^ (lx >> 1);
    h ^= h >> 13; h ^= h << 7; h ^= h >> 3;
    return h & mask;
}

/* ── saturating uint16 add ───────────────────────────────────────────────── */
static uint16_t sat16(uint16_t a, uint16_t b) {
    uint32_t s = (uint32_t)a + b;
    return s > 0xFFFFU ? 0xFFFFU : (uint16_t)s;
}

/* ── bucket operations ───────────────────────────────────────────────────── */
static void bucket_inc(uint16_t *tok_arr, uint16_t *cnt_arr,
                       uint32_t slots, uint32_t ctx, uint16_t tok) {
    uint32_t base = ctx * slots;
    int free_i = -1, min_i = 0;
    uint16_t min_c = 0xFFFFU;

    for (uint32_t i = 0; i < slots; ++i) {
        if (cnt_arr[base + i] == 0U) { if (free_i < 0) free_i = (int)i; continue; }
        if (tok_arr[base + i] == tok) { cnt_arr[base + i] = sat16(cnt_arr[base + i], 1U); return; }
        if (cnt_arr[base + i] < min_c) { min_c = cnt_arr[base + i]; min_i = (int)i; }
    }
    if (free_i >= 0) { tok_arr[base + free_i] = tok; cnt_arr[base + free_i] = 1U; return; }
    if (cnt_arr[base + min_i] > 1U) cnt_arr[base + min_i]--;
    else { tok_arr[base + min_i] = tok; cnt_arr[base + min_i] = 1U; }
}

static int bucket_pred(const uint16_t *tok_arr, const uint16_t *cnt_arr,
                       uint32_t slots, uint32_t ctx,
                       uint16_t *tok_out, uint32_t *tot_out, uint32_t *best_out) {
    uint32_t base = ctx * slots;
    uint32_t total = 0U, best = 0U;
    uint16_t best_tok = 0U;
    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t c = cnt_arr[base + i];
        total += c;
        if (c > best) { best = c; best_tok = tok_arr[base + i]; }
    }
    if (total == 0U) return 0;
    *tok_out = best_tok; *tot_out = total; *best_out = best;
    return 1;
}

/* Integer weighted random sampling (temp_shift: 0=greedy, 1=linear, 2=count^2) */
static uint16_t bucket_sample(const uint16_t *tok_arr, const uint16_t *cnt_arr,
                               uint32_t slots, uint32_t ctx,
                               uint32_t *rng, uint32_t temp) {
    uint32_t base = ctx * slots;
    uint32_t total = 0U, best = 0U;
    uint16_t best_tok = 0U;

    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t c = cnt_arr[base + i];
        if (!c) continue;
        uint32_t w = c;
        for (uint32_t s = 0; s < temp; s++) {
            uint64_t t2 = (uint64_t)w * c;
            w = t2 > 0xFFFFUL ? 0xFFFFU : (uint32_t)t2;
        }
        total += w;
        if (c > best) { best = c; best_tok = tok_arr[base + i]; }
    }
    if (!total || !temp) return best_tok;

    uint32_t x = *rng; x ^= x<<13; x ^= x>>17; x ^= x<<5; *rng = x;
    uint32_t pick = x % total;
    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t c = cnt_arr[base + i];
        if (!c) continue;
        uint32_t w = c;
        for (uint32_t s = 0; s < temp; s++) {
            uint64_t t2 = (uint64_t)w * c;
            w = t2 > 0xFFFFUL ? 0xFFFFU : (uint32_t)t2;
        }
        if (pick < w) return tok_arr[base + i];
        pick -= w;
    }
    return best_tok;
}

/* ── confidence score (pure integer) ────────────────────────────────────── */
static uint32_t conf_score(uint32_t best, uint32_t total) {
    if (!total) return 0U;
    return (best * 0xFFFFU) / (total + CONF_SMOOTH);
}

/* ── model alloc / free ──────────────────────────────────────────────────── */
static int alloc_tables(word_lm_t *lm) {
    size_t n = (size_t)lm->ctx_slots * lm->slots;
    lm->vocab    = calloc(VOCAB_SIZE, WORD_MAXLEN);
    lm->unigram  = calloc(VOCAB_SIZE, sizeof(uint32_t));
    lm->bi_tok   = calloc(n, sizeof(uint16_t));
    lm->bi_cnt   = calloc(n, sizeof(uint16_t));
    lm->tri_tok  = calloc(n, sizeof(uint16_t));
    lm->tri_cnt  = calloc(n, sizeof(uint16_t));
    lm->quad_tok = calloc(n, sizeof(uint16_t));
    lm->quad_cnt = calloc(n, sizeof(uint16_t));
    return lm->vocab && lm->unigram
        && lm->bi_tok && lm->bi_cnt && lm->tri_tok && lm->tri_cnt
        && lm->quad_tok && lm->quad_cnt;
}
static void free_tables(word_lm_t *lm) {
    free(lm->vocab);    lm->vocab = NULL;
    free(lm->unigram);  lm->unigram = NULL;
    free(lm->bi_tok);   free(lm->bi_cnt);
    free(lm->tri_tok);  free(lm->tri_cnt);
    free(lm->quad_tok); free(lm->quad_cnt);
}

/* ── model save / load ───────────────────────────────────────────────────── */
static int save_model(const word_lm_t *lm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }

    uint32_t hdr[5] = { WORD_MAGIC, WORD_VERSION, VOCAB_SIZE, lm->ctx_slots, lm->slots };
    fwrite(hdr, 4, 5, f);
    fwrite(lm->vocab[0], WORD_MAXLEN, VOCAB_SIZE, f);
    fwrite(lm->unigram,   4,           VOCAB_SIZE, f);

    size_t n = (size_t)lm->ctx_slots * lm->slots;
    fwrite(lm->bi_tok,   2, n, f);
    fwrite(lm->bi_cnt,   2, n, f);
    fwrite(lm->tri_tok,  2, n, f);
    fwrite(lm->tri_cnt,  2, n, f);
    fwrite(lm->quad_tok, 2, n, f);
    fwrite(lm->quad_cnt, 2, n, f);
    fclose(f);
    return 1;
}

static int load_model(word_lm_t *lm, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }

    uint32_t hdr[5];
    if (fread(hdr, 4, 5, f) != 5 || hdr[0] != WORD_MAGIC || hdr[1] != WORD_VERSION) {
        fprintf(stderr, "Bad magic/version in %s\n", path); fclose(f); return 0;
    }
    lm->ctx_slots = hdr[3];
    lm->slots     = hdr[4];
    lm->mask      = lm->ctx_slots - 1U;
    lm->lxor      = 0U;
    memset(lm->ctx_ring, 0, sizeof(lm->ctx_ring));
    lm->ctx_head  = 0U;

    if (!alloc_tables(lm)) { fclose(f); return 0; }

    fread(lm->vocab[0], WORD_MAXLEN, VOCAB_SIZE, f);
    fread(lm->unigram,  4,           VOCAB_SIZE, f);

    size_t n = (size_t)lm->ctx_slots * lm->slots;
    fread(lm->bi_tok,   2, n, f);
    fread(lm->bi_cnt,   2, n, f);
    fread(lm->tri_tok,  2, n, f);
    fread(lm->tri_cnt,  2, n, f);
    fread(lm->quad_tok, 2, n, f);
    fread(lm->quad_cnt, 2, n, f);
    fclose(f);
    return 1;
}

/* ── training kernel ─────────────────────────────────────────────────────── */
static void wlm_update(word_lm_t *lm, uint16_t wid) {
    uint16_t t1 = recent_word(lm, 0);
    uint16_t t2 = recent_word(lm, 1);
    uint16_t t3 = recent_word(lm, 2);

    uint32_t bi   = h_bi(t1, lm->mask);
    uint32_t tri  = h_tri(t2, t1, lm->mask);
    uint32_t quad = h_quad(t3, t2, t1, lm->mask);

    lm->unigram[wid]++;
    bucket_inc(lm->bi_tok,   lm->bi_cnt,   lm->slots, bi,   wid);
    bucket_inc(lm->tri_tok,  lm->tri_cnt,  lm->slots, tri,  wid);
    bucket_inc(lm->quad_tok, lm->quad_cnt, lm->slots, quad, wid);

    lm->lxor = lxor_upd(lm->lxor, wid);
    push_word(lm, wid);
}

/* ── prediction ──────────────────────────────────────────────────────────── */
static uint16_t wlm_predict(const word_lm_t *lm) {
    uint16_t t1 = recent_word(lm, 0);
    uint16_t t2 = recent_word(lm, 1);
    uint16_t t3 = recent_word(lm, 2);

    uint32_t bi   = h_bi(t1, lm->mask);
    uint32_t tri  = h_tri(t2, t1, lm->mask);
    uint32_t quad = h_quad(t3, t2, t1, lm->mask);

    uint16_t tok; uint32_t total, best;
    uint32_t best_sc = 0U; uint16_t best_tok = 0U;

    if (bucket_pred(lm->quad_tok, lm->quad_cnt, lm->slots, quad, &tok, &total, &best)) {
        uint32_t sc = conf_score(best, total);
        if (sc > best_sc) { best_sc = sc; best_tok = tok; }
        if (best_sc >= CONF_ENOUGH) return best_tok;
    }
    if (bucket_pred(lm->tri_tok, lm->tri_cnt, lm->slots, tri, &tok, &total, &best)) {
        uint32_t sc = conf_score(best, total);
        if (sc > best_sc) { best_sc = sc; best_tok = tok; }
        if (best_sc >= CONF_ENOUGH) return best_tok;
    }
    if (bucket_pred(lm->bi_tok, lm->bi_cnt, lm->slots, bi, &tok, &total, &best)) {
        uint32_t sc = conf_score(best, total);
        if (sc > best_sc) { best_sc = sc; best_tok = tok; }
    }
    if (best_sc) return best_tok;

    /* unigram fallback: most frequent word */
    uint32_t ub = 0U; uint16_t ut = 0U;
    for (uint32_t i = 0; i < VOCAB_SIZE; i++) {
        if (lm->unigram[i] > ub) { ub = lm->unigram[i]; ut = (uint16_t)i; }
    }
    return ut;
}

static uint16_t wlm_sample(word_lm_t *lm, uint32_t *rng, uint32_t temp) {
    uint16_t t1 = recent_word(lm, 0);
    uint16_t t2 = recent_word(lm, 1);
    uint16_t t3 = recent_word(lm, 2);

    uint32_t bi   = h_bi(t1, lm->mask);
    uint32_t tri  = h_tri(t2, t1, lm->mask);
    uint32_t quad = h_quad(t3, t2, t1, lm->mask);

    uint16_t tok; uint32_t total, best;

    if (bucket_pred(lm->quad_tok, lm->quad_cnt, lm->slots, quad, &tok, &total, &best))
        return bucket_sample(lm->quad_tok, lm->quad_cnt, lm->slots, quad, rng, temp);
    if (bucket_pred(lm->tri_tok, lm->tri_cnt, lm->slots, tri, &tok, &total, &best))
        return bucket_sample(lm->tri_tok,  lm->tri_cnt,  lm->slots, tri,  rng, temp);
    if (bucket_pred(lm->bi_tok, lm->bi_cnt, lm->slots, bi, &tok, &total, &best))
        return bucket_sample(lm->bi_tok,   lm->bi_cnt,   lm->slots, bi,   rng, temp);

    /* unigram sample */
    uint32_t ug_total = 0U;
    for (uint32_t i = 0; i < VOCAB_SIZE; i++) ug_total += lm->unigram[i];
    if (ug_total) {
        uint32_t x = *rng; x^=x<<13; x^=x>>17; x^=x<<5; *rng=x;
        uint32_t pick = x % ug_total;
        for (uint32_t i = 0; i < VOCAB_SIZE; i++) {
            if (pick < lm->unigram[i]) return (uint16_t)i;
            pick -= lm->unigram[i];
        }
    }
    return 0U;
}

/* ── corpus streaming tokeniser ─────────────────────────────────────────── */
/* Process one character: if a word boundary, flush; else accumulate.
 * Returns word ID if word completed, 0xFFFFFFFF otherwise. */
typedef struct {
    char buf[64];
    uint32_t len;
} WordBuf;

/* Process one character: accumulate word chars, flush on boundary.
 * lm: model (for vocab access). insert=1 to add new words, 0 for lookup only.
 * Returns 1 + sets *wid_out when a word is completed; 0 otherwise. */
static uint32_t wb_push(WordBuf *wb, int c, word_lm_t *lm, int insert, uint16_t *wid_out) {
    char lc;
    if (word_char(c, &lc)) {
        if (wb->len < 63U) wb->buf[wb->len++] = lc;
        return 0;
    }
    if (wb->len == 0U) return 0;
    wb->buf[wb->len] = '\0';
    uint16_t wid = vocab_id(lm->vocab, wb->buf, wb->len, insert);
    wb->len = 0;
    if (wid == 0xFFFFU) return 0; /* table full or word not in vocab */
    if (wid_out) *wid_out = wid;
    return 1;
}

/* Train on one open FILE, one pass */
static uint64_t train_file_one_pass(word_lm_t *lm, FILE *f, int online_update) {
    static char chunk[CHUNK_SZ];
    WordBuf wb; wb.len = 0;
    uint64_t words = 0;
    size_t nr;
    (void)online_update;

    while ((nr = fread(chunk, 1, CHUNK_SZ, f)) > 0) {
        for (size_t i = 0; i < nr; i++) {
            uint16_t wid;
            if (wb_push(&wb, (unsigned char)chunk[i], lm, 1, &wid)) {
                wlm_update(lm, wid);
                words++;
            }
        }
    }
    /* flush last word */
    uint16_t wid;
    if (wb_push(&wb, ' ', lm, 1, &wid)) {
        wlm_update(lm, wid);
        words++;
    }
    return words;
}

/* ── commands ────────────────────────────────────────────────────────────── */
static void usage(const char *exe) {
    printf("Usage:\n");
    printf("  %s train      <corpus> <model.wlm> [passes=4] [ctx=65536] [slots=16]\n", exe);
    printf("  %s resume     <model.wlm> <corpus> [passes=4]\n", exe);
    printf("  %s train-list <list> <model.wlm> [passes=4] [ctx=65536] [slots=16]\n", exe);
    printf("  %s eval       <model.wlm> <corpus>\n", exe);
    printf("  %s gen        <model.wlm> <prompt> [max_words=64]\n", exe);
    printf("  %s gen-sample <model.wlm> <prompt> [max_words=64] [temp=1] [seed=0]\n", exe);
    printf("Notes:\n");
    printf("  temp: 0=greedy 1=count-weighted 2=count^2-weighted (all integer, no floats)\n");
    printf("  vocab: 65536 16-bit word IDs via FNV-1a, tables over word sequences\n");
}

static int cmd_train(int argc, char **argv) {
    const char *corpus_path = argv[2];
    const char *model_path  = argv[3];
    uint32_t passes    = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 4U;
    uint32_t ctx_slots = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : DEFAULT_CTX;
    uint32_t slots     = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : DEFAULT_SLOTS;

    /* ctx_slots must be power of 2 */
    uint32_t p2 = 1U;
    while (p2 < ctx_slots) p2 <<= 1;
    ctx_slots = p2;

    word_lm_t lm;
    memset(&lm, 0, sizeof(lm));
    lm.slots     = slots;
    lm.ctx_slots = ctx_slots;
    lm.mask      = ctx_slots - 1U;
    lm.lxor      = 0U;

    if (!alloc_tables(&lm)) { fprintf(stderr, "OOM\n"); return 1; }

    printf("Training word LM: corpus=%s model=%s passes=%u ctx=%u slots=%u\n",
           corpus_path, model_path, passes, ctx_slots, slots);

    for (uint32_t p = 0; p < passes; p++) {
        FILE *f = fopen(corpus_path, "rb");
        if (!f) { perror(corpus_path); free_tables(&lm); return 1; }
        /* reset context between passes */
        memset(lm.ctx_ring, 0, sizeof(lm.ctx_ring));
        lm.ctx_head = 0; lm.lxor = 0;
        uint64_t words = train_file_one_pass(&lm, f, 0);
        fclose(f);
        printf("  pass %u: %llu words\n", p+1, (unsigned long long)words);
    }

    if (!save_model(&lm, model_path)) { free_tables(&lm); return 1; }
    printf("Saved: %s\n", model_path);
    free_tables(&lm);
    return 0;
}

static int cmd_resume(int argc, char **argv) {
    const char *model_path  = argv[2];
    const char *corpus_path = argv[3];
    uint32_t passes = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 4U;

    word_lm_t lm;
    if (!load_model(&lm, model_path)) return 1;

    for (uint32_t p = 0; p < passes; p++) {
        FILE *f = fopen(corpus_path, "rb");
        if (!f) { perror(corpus_path); free_tables(&lm); return 1; }
        memset(lm.ctx_ring, 0, sizeof(lm.ctx_ring));
        lm.ctx_head = 0; lm.lxor = 0;
        uint64_t words = train_file_one_pass(&lm, f, 0);
        fclose(f);
        printf("  resume pass %u: %llu words\n", p+1, (unsigned long long)words);
    }

    if (!save_model(&lm, model_path)) { free_tables(&lm); return 1; }
    printf("Saved: %s\n", model_path);
    free_tables(&lm);
    return 0;
}

static int cmd_train_list(int argc, char **argv) {
    const char *list_path  = argv[2];
    const char *model_path = argv[3];
    uint32_t passes    = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 4U;
    uint32_t ctx_slots = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : DEFAULT_CTX;
    uint32_t slots     = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : DEFAULT_SLOTS;

    uint32_t p2 = 1U;
    while (p2 < ctx_slots) p2 <<= 1;
    ctx_slots = p2;

    word_lm_t lm;
    memset(&lm, 0, sizeof(lm));
    lm.slots = slots; lm.ctx_slots = ctx_slots; lm.mask = ctx_slots - 1U;
    if (!alloc_tables(&lm)) { fprintf(stderr, "OOM\n"); return 1; }

    for (uint32_t p = 0; p < passes; p++) {
        FILE *lf = fopen(list_path, "r");
        if (!lf) { perror(list_path); free_tables(&lm); return 1; }
        char line[2048]; uint64_t total_words = 0;
        while (fgets(line, sizeof(line), lf)) {
            /* strip BOM and whitespace */
            char *cur = line;
            if ((unsigned char)cur[0]==0xEF && (unsigned char)cur[1]==0xBB && (unsigned char)cur[2]==0xBF) cur+=3;
            size_t l = strlen(cur);
            while (l && (cur[l-1]=='\n'||cur[l-1]=='\r'||cur[l-1]==' ')) cur[--l]='\0';
            if (!l || cur[0]=='#') continue;

            FILE *f = fopen(cur, "rb");
            if (!f) { fprintf(stderr,"Cannot open: %s\n",cur); continue; }
            memset(lm.ctx_ring, 0, sizeof(lm.ctx_ring)); lm.ctx_head=0; lm.lxor=0;
            total_words += train_file_one_pass(&lm, f, 0);
            fclose(f);
        }
        fclose(lf);
        printf("  pass %u: %llu words\n", p+1, (unsigned long long)total_words);
    }
    if (!save_model(&lm, model_path)) { free_tables(&lm); return 1; }
    printf("Saved: %s\n", model_path);
    free_tables(&lm);
    return 0;
}

static int cmd_eval(int argc, char **argv) {
    const char *model_path  = argv[2];
    const char *corpus_path = (argc > 3) ? argv[3] : NULL;
    (void)argc;

    word_lm_t lm;
    if (!load_model(&lm, model_path)) return 1;
    word_lm_t *lm_ptr = &lm;

    FILE *f = corpus_path ? fopen(corpus_path, "rb") : stdin;
    if (!f) { perror(corpus_path); free_tables(&lm); return 1; }

    uint64_t correct = 0, total = 0;
    static char chunk[CHUNK_SZ];
    WordBuf wb; wb.len = 0;
    size_t nr;

    while ((nr = fread(chunk, 1, CHUNK_SZ, f)) > 0) {
        for (size_t i = 0; i < nr; i++) {
            uint16_t obs;
            if (!wb_push(&wb, (unsigned char)chunk[i], lm_ptr, 0, &obs)) continue;
            uint16_t pred = wlm_predict(&lm);
            if (pred == obs) correct++;
            total++;
            wlm_update(&lm, obs);
        }
    }
    /* flush */
    uint16_t obs;
    if (wb_push(&wb, ' ', lm_ptr, 0, &obs)) {
        uint16_t pred = wlm_predict(&lm);
        if (pred == obs) correct++;
        total++;
        wlm_update(&lm, obs);
    }
    if (corpus_path) fclose(f);

    double acc = total ? (double)correct / (double)total * 100.0 : 0.0;
    /* print as integer × 100 to avoid floats in hot path, but printf here is fine */
    printf("words: %llu  correct: %llu  accuracy: %.2f%%\n",
           (unsigned long long)total, (unsigned long long)correct, acc);
    free_tables(&lm);
    return 0;
}

/* Update context ring + lxor only — no training. */
static void wlm_ctx_only(word_lm_t *lm, uint16_t wid) {
    lm->lxor = lxor_upd(lm->lxor, wid);
    push_word(lm, wid);
}

/* Tokenise prompt: lookup word IDs (no insert), update context only. */
static void prime_ctx(word_lm_t *lm, const char *prompt) {
    WordBuf wb; wb.len = 0;
    for (const char *p = prompt; *p; p++) {
        uint16_t wid;
        if (wb_push(&wb, (unsigned char)*p, lm, 0, &wid))
            wlm_ctx_only(lm, wid);
    }
    uint16_t wid;
    if (wb_push(&wb, ' ', lm, 0, &wid))
        wlm_ctx_only(lm, wid);
}

static void emit_words(word_lm_t *lm, uint32_t max_words, uint32_t temp, uint32_t seed) {
    uint32_t rng = seed ? seed : 0xDEADBEEFU;
    uint16_t prev = 0xFFFFU;
    uint32_t run  = 0U;

    /* short history for phrase-loop detection */
    uint16_t hist[32];
    uint32_t hlen = 0;

    for (uint32_t w = 0; w < max_words; w++) {
        uint16_t wid;
        if (temp == 0U) wid = wlm_predict(lm);
        else            wid = wlm_sample(lm, &rng, temp);

        /* single-word repetition guard */
        if (wid == prev) run++; else run = 0U;
        prev = wid;
        if (run > 3U && w > 4U) break;

        /* phrase-loop detection: check if last 3 words appeared earlier */
        if (hlen >= 6U) {
            uint32_t wlen = 3U;
            int looped = 0;
            for (uint32_t j = 0; j + wlen <= hlen - wlen; ++j) {
                if (hist[hlen - wlen] == hist[j]
                    && hist[hlen - wlen + 1] == hist[j + 1]
                    && hist[hlen - wlen + 2] == hist[j + 2]) {
                    looped = 1; break;
                }
            }
            if (looped) break;
        }

        const char *word = lm->vocab[wid];
        if (word[0] == '\0') printf("<unk> ");
        else printf("%s ", word);

        lm->lxor = lxor_upd(lm->lxor, wid);
        push_word(lm, wid);

        if (hlen < 32U) hist[hlen++] = wid;
        else { memmove(hist, hist+1, 31*sizeof(uint16_t)); hist[31] = wid; }
    }
    putchar('\n');
}

static int cmd_gen(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *prompt     = argv[3];
    uint32_t max_words = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 64U;

    word_lm_t lm;
    if (!load_model(&lm, model_path)) return 1;
    memset(lm.ctx_ring, 0, sizeof(lm.ctx_ring)); lm.ctx_head=0; lm.lxor=0;
    printf("%s ", prompt);
    prime_ctx(&lm, prompt);
    emit_words(&lm, max_words, 0U, 0U);
    free_tables(&lm);
    return 0;
}

static int cmd_gen_sample(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *prompt     = argv[3];
    uint32_t max_words = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 64U;
    uint32_t temp      = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : 1U;
    uint32_t seed      = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : (uint32_t)time(NULL);
    if (temp > 4U) temp = 4U;

    word_lm_t lm;
    if (!load_model(&lm, model_path)) return 1;
    memset(lm.ctx_ring, 0, sizeof(lm.ctx_ring)); lm.ctx_head=0; lm.lxor=0;
    printf("%s ", prompt);
    prime_ctx(&lm, prompt);
    emit_words(&lm, max_words, temp, seed);
    free_tables(&lm);
    return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    if      (strcmp(argv[1], "train")      == 0 && argc >= 4) return cmd_train(argc, argv);
    else if (strcmp(argv[1], "resume")     == 0 && argc >= 4) return cmd_resume(argc, argv);
    else if (strcmp(argv[1], "train-list") == 0 && argc >= 4) return cmd_train_list(argc, argv);
    else if (strcmp(argv[1], "eval")       == 0 && argc >= 3) return cmd_eval(argc, argv);
    else if (strcmp(argv[1], "gen")        == 0 && argc >= 4) return cmd_gen(argc, argv);
    else if (strcmp(argv[1], "gen-sample") == 0 && argc >= 4) return cmd_gen_sample(argc, argv);

    usage(argv[0]);
    return 1;
}
