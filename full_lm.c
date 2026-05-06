#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define VOCAB_SIZE 256U
#define WORK_RING  1024U
#define CONF_SMOOTH 4U
#define CONF_ENOUGH 0xB000U
#define MAGIC 0x314D4C4EU /* NLM1 */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t slots;
    uint32_t ctx_count;
    uint32_t reserved;
} model_header_t;

typedef struct {
    uint32_t slots;
    uint32_t ctx_count;
    uint32_t mask;

    uint32_t *unigram;          /* [256] */
    uint16_t *bigram;           /* [256*256] */

    uint8_t  *tri_tok;          /* [ctx*slots] */
    uint16_t *tri_cnt;          /* [ctx*slots] */
    uint8_t  *quad_tok;
    uint16_t *quad_cnt;
    uint8_t  *oct_tok;
    uint16_t *oct_cnt;
    uint8_t  *ent_tok;
    uint16_t *ent_cnt;

    uint8_t  work[WORK_RING];
    uint32_t work_head;
    uint32_t lxor;
} fused_lm_t;

static inline uint16_t sat16(uint16_t v, uint16_t a) {
    uint32_t n = (uint32_t)v + a;
    return n > 65535U ? 65535U : (uint16_t)n;
}

static inline uint8_t recent(const fused_lm_t *lm, uint32_t back) {
    return lm->work[(lm->work_head + WORK_RING - 1U - back) % WORK_RING];
}

static inline void push_ctx(fused_lm_t *lm, uint8_t tok) {
    lm->work[lm->work_head] = tok;
    lm->work_head = (lm->work_head + 1U) % WORK_RING;
}

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
    uint32_t h = ((uint32_t)t7 << 24) | ((uint32_t)t6 << 16) | ((uint32_t)t5 << 8) | (uint32_t)t4;
    h ^= ((uint32_t)t3 << 16) | ((uint32_t)t2 << 8) | (uint32_t)t1;
    h ^= h << 5; h ^= h >> 11; h ^= h << 13;
    return h;
}

static inline uint16_t h_ent(uint8_t a, uint8_t b, uint8_t c, uint32_t lxor, uint32_t mask) {
    uint32_t s = h_quad(a, b, c, 0xFFFFU);
    uint32_t l = lxor ^ (lxor >> 16);
    return (uint16_t)((s ^ l) & mask);
}

static inline uint32_t lxor_upd(uint32_t acc, uint8_t tok) {
    return ((acc << 1) | (acc >> 31)) ^ (uint32_t)tok;
}

static inline uint32_t conf_score(uint32_t best, uint32_t total) {
    return (best << 16) / (total + CONF_SMOOTH);
}

#define BG(lm,a,b) ((lm)->bigram[(uint32_t)(a)*VOCAB_SIZE + (uint32_t)(b)])

static int alloc_tables(fused_lm_t *lm, uint32_t slots, uint32_t ctx_count) {
    uint64_t nslots;

    memset(lm, 0, sizeof(*lm));
    if (!ctx_count || (ctx_count & (ctx_count - 1U))) return 0;

    nslots = (uint64_t)slots * (uint64_t)ctx_count;
    if (nslots == 0U || nslots > (uint64_t)0x7fffffffU) return 0;

    lm->slots = slots;
    lm->ctx_count = ctx_count;
    lm->mask = ctx_count - 1U;

    lm->unigram = (uint32_t *)calloc(VOCAB_SIZE, sizeof(uint32_t));
    lm->bigram = (uint16_t *)calloc(VOCAB_SIZE * VOCAB_SIZE, sizeof(uint16_t));

    lm->tri_tok = (uint8_t *)calloc((size_t)nslots, sizeof(uint8_t));
    lm->tri_cnt = (uint16_t *)calloc((size_t)nslots, sizeof(uint16_t));
    lm->quad_tok = (uint8_t *)calloc((size_t)nslots, sizeof(uint8_t));
    lm->quad_cnt = (uint16_t *)calloc((size_t)nslots, sizeof(uint16_t));
    lm->oct_tok = (uint8_t *)calloc((size_t)nslots, sizeof(uint8_t));
    lm->oct_cnt = (uint16_t *)calloc((size_t)nslots, sizeof(uint16_t));
    lm->ent_tok = (uint8_t *)calloc((size_t)nslots, sizeof(uint8_t));
    lm->ent_cnt = (uint16_t *)calloc((size_t)nslots, sizeof(uint16_t));

    if (!lm->unigram || !lm->bigram ||
        !lm->tri_tok || !lm->tri_cnt ||
        !lm->quad_tok || !lm->quad_cnt ||
        !lm->oct_tok || !lm->oct_cnt ||
        !lm->ent_tok || !lm->ent_cnt) {
        return 0;
    }
    return 1;
}

static void free_tables(fused_lm_t *lm) {
    free(lm->unigram);
    free(lm->bigram);
    free(lm->tri_tok); free(lm->tri_cnt);
    free(lm->quad_tok); free(lm->quad_cnt);
    free(lm->oct_tok); free(lm->oct_cnt);
    free(lm->ent_tok); free(lm->ent_cnt);
    memset(lm, 0, sizeof(*lm));
}

static void bucket_inc(uint8_t *tok_arr, uint16_t *cnt_arr, uint32_t slots, uint32_t ctx, uint8_t tok) {
    uint32_t base = ctx * slots;
    int free_i = -1;
    uint32_t min_i = 0;
    uint16_t min_c = cnt_arr[base + 0];

    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t idx = base + i;
        uint16_t c = cnt_arr[idx];

        if (c == 0U && free_i < 0) free_i = (int)i;
        if (c && tok_arr[idx] == tok) {
            cnt_arr[idx] = sat16(c, 1U);
            return;
        }
        if (c < min_c) {
            min_c = c;
            min_i = i;
        }
    }

    if (free_i >= 0) {
        uint32_t idx = base + (uint32_t)free_i;
        tok_arr[idx] = tok;
        cnt_arr[idx] = 1U;
        return;
    }

    {
        uint32_t idx = base + min_i;
        if (cnt_arr[idx] > 1U) cnt_arr[idx]--;
        else {
            tok_arr[idx] = tok;
            cnt_arr[idx] = 1U;
        }
    }
}

static int bucket_pred(const uint8_t *tok_arr, const uint16_t *cnt_arr,
                       uint32_t slots, uint32_t ctx,
                       uint8_t *tok_out, uint32_t *tot_out, uint32_t *best_out) {
    uint32_t base = ctx * slots;
    uint32_t total = 0U;
    uint32_t best = 0U;
    uint8_t best_tok = 32U;

    for (uint32_t i = 0; i < slots; ++i) {
        uint16_t c = cnt_arr[base + i];
        total += c;
        if (c > best) {
            best = c;
            best_tok = tok_arr[base + i];
        }
    }

    if (total == 0U) return 0;
    *tok_out = best_tok;
    *tot_out = total;
    *best_out = best;
    return 1;
}

/* Integer weighted random sampling — no floats.
 * temp_shift: 0 = greedy (max count), 1 = linear weights, 2 = squared weights.
 * rng_state: xorshift32 state (updated in place).
 * Returns a token sampled proportional to count^(1<<temp_shift). */
static uint8_t bucket_sample(const uint8_t *tok_arr, const uint16_t *cnt_arr,
                              uint32_t slots, uint32_t ctx,
                              uint32_t *rng_state, uint32_t temp_shift) {
    uint32_t base = ctx * slots;
    uint32_t total = 0U;
    uint8_t best_tok = 32U;
    uint32_t best_cnt = 0U;

    /* First pass: compute total weight and find greedy best */
    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t c = cnt_arr[base + i];
        if (c == 0U) continue;
        /* Weight = c raised to 2^temp_shift (integer shifts) */
        uint32_t w = c;
        for (uint32_t s = 0; s < temp_shift; s++) {
            /* w *= c but clamp at 0xFFFFU to avoid overflow */
            uint64_t tmp = (uint64_t)w * c;
            w = tmp > 0xFFFFUL ? 0xFFFFU : (uint32_t)tmp;
        }
        total += w;
        if (c > best_cnt) { best_cnt = c; best_tok = tok_arr[base + i]; }
    }

    if (total == 0U) return best_tok;

    /* temp_shift==0 means greedy */
    if (temp_shift == 0U) return best_tok;

    /* xorshift32 random pick in [0, total) */
    uint32_t x = *rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *rng_state = x;
    uint32_t pick = x % total;

    /* Second pass: walk candidates */
    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t c = cnt_arr[base + i];
        if (c == 0U) continue;
        uint32_t w = c;
        for (uint32_t s = 0; s < temp_shift; s++) {
            uint64_t tmp = (uint64_t)w * c;
            w = tmp > 0xFFFFUL ? 0xFFFFU : (uint32_t)tmp;
        }
        if (pick < w) return tok_arr[base + i];
        pick -= w;
    }
    return best_tok;
}

static uint8_t bigram_pred(const fused_lm_t *lm, uint8_t t1, uint8_t *conf_out) {
    uint32_t total = 0U;
    uint32_t best = 0U;
    uint8_t best_tok = 32U;

    for (uint32_t i = 0; i < VOCAB_SIZE; ++i) {
        uint16_t c = BG(lm, t1, i);
        total += c;
        if (c > best) {
            best = c;
            best_tok = (uint8_t)i;
        }
    }
    *conf_out = total ? (uint8_t)((best * 255U) / total) : 0U;
    return best_tok;
}

static void lm_update(fused_lm_t *lm, uint8_t obs) {
    uint8_t t1 = recent(lm, 0), t2 = recent(lm, 1), t3 = recent(lm, 2);
    uint8_t t4 = recent(lm, 3), t5 = recent(lm, 4), t6 = recent(lm, 5), t7 = recent(lm, 6);

    uint16_t tri = h_tri(t2, t1, lm->mask);
    uint16_t quad = h_quad(t3, t2, t1, lm->mask);
    uint16_t oct = (uint16_t)(h_oct7(t7, t6, t5, t4, t3, t2, t1) & lm->mask);
    uint16_t ent = h_ent(t3, t2, t1, lm->lxor, lm->mask);

    lm->unigram[obs]++;
    BG(lm, t1, obs) = sat16(BG(lm, t1, obs), 1U);

    bucket_inc(lm->tri_tok, lm->tri_cnt, lm->slots, tri, obs);
    bucket_inc(lm->quad_tok, lm->quad_cnt, lm->slots, quad, obs);
    bucket_inc(lm->oct_tok, lm->oct_cnt, lm->slots, oct, obs);
    bucket_inc(lm->ent_tok, lm->ent_cnt, lm->slots, ent, obs);

    lm->lxor = lxor_upd(lm->lxor, obs);
    push_ctx(lm, obs);
}

static uint8_t lm_predict(const fused_lm_t *lm, uint8_t *conf_out) {
    uint8_t t1 = recent(lm, 0), t2 = recent(lm, 1), t3 = recent(lm, 2);
    uint8_t t4 = recent(lm, 3), t5 = recent(lm, 4), t6 = recent(lm, 5), t7 = recent(lm, 6);

    uint16_t tri = h_tri(t2, t1, lm->mask);
    uint16_t quad = h_quad(t3, t2, t1, lm->mask);
    uint16_t oct = (uint16_t)(h_oct7(t7, t6, t5, t4, t3, t2, t1) & lm->mask);
    uint16_t ent = h_ent(t3, t2, t1, lm->lxor, lm->mask);

    uint8_t tok, best_tok = 32U, best_co = 0U;
    uint32_t total, best, sc, best_sc = 0U;

    if (bucket_pred(lm->oct_tok, lm->oct_cnt, lm->slots, oct, &tok, &total, &best)) {
        sc = conf_score(best, total);
        if (sc > best_sc) {
            best_sc = sc;
            best_tok = tok;
            best_co = (uint8_t)((best * 255U) / total);
        }
    }
    if (best_sc >= CONF_ENOUGH) {
        *conf_out = best_co;
        return best_tok;
    }

    if (bucket_pred(lm->ent_tok, lm->ent_cnt, lm->slots, ent, &tok, &total, &best)) {
        sc = conf_score(best, total);
        if (sc > best_sc) {
            best_sc = sc;
            best_tok = tok;
            best_co = (uint8_t)((best * 255U) / total);
        }
    }
    if (best_sc >= CONF_ENOUGH) {
        *conf_out = best_co;
        return best_tok;
    }

    if (bucket_pred(lm->quad_tok, lm->quad_cnt, lm->slots, quad, &tok, &total, &best)) {
        sc = conf_score(best, total);
        if (sc > best_sc) {
            best_sc = sc;
            best_tok = tok;
            best_co = (uint8_t)((best * 255U) / total);
        }
    }
    if (best_sc >= CONF_ENOUGH) {
        *conf_out = best_co;
        return best_tok;
    }

    if (bucket_pred(lm->tri_tok, lm->tri_cnt, lm->slots, tri, &tok, &total, &best)) {
        sc = conf_score(best, total);
        if (sc > best_sc) {
            best_sc = sc;
            best_tok = tok;
            best_co = (uint8_t)((best * 255U) / total);
        }
    }

    if (best_sc) {
        *conf_out = best_co;
        return best_tok;
    }

    return bigram_pred(lm, t1, conf_out);
}

static void lm_prime_ctx(fused_lm_t *lm, const uint8_t *text, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {
        lm->lxor = lxor_upd(lm->lxor, text[i]);
        push_ctx(lm, text[i]);
    }
}

/* Sample next token using integer weighted random (temp_shift=0: greedy, 1: linear, 2: squared).
 * Falls through the same confidence cascade as lm_predict, but samples instead of max. */
static uint8_t lm_sample(fused_lm_t *lm, uint32_t *rng, uint32_t temp_shift) {
    uint8_t t1 = recent(lm, 0), t2 = recent(lm, 1), t3 = recent(lm, 2);
    uint8_t t4 = recent(lm, 3), t5 = recent(lm, 4), t6 = recent(lm, 5), t7 = recent(lm, 6);

    uint16_t tri  = h_tri(t2, t1, lm->mask);
    uint16_t quad = h_quad(t3, t2, t1, lm->mask);
    uint16_t oct  = (uint16_t)(h_oct7(t7,t6,t5,t4,t3,t2,t1) & lm->mask);
    uint16_t ent  = h_ent(t3, t2, t1, lm->lxor, lm->mask);

    /* Try highest-order tables first; fall back if empty */
    uint8_t tok;
    uint32_t total = 0U, best = 0U;

    /* oct */
    if (bucket_pred(lm->oct_tok, lm->oct_cnt, lm->slots, oct, &tok, &total, &best))
        return bucket_sample(lm->oct_tok, lm->oct_cnt, lm->slots, oct, rng, temp_shift);
    /* ent */
    if (bucket_pred(lm->ent_tok, lm->ent_cnt, lm->slots, ent, &tok, &total, &best))
        return bucket_sample(lm->ent_tok, lm->ent_cnt, lm->slots, ent, rng, temp_shift);
    /* quad */
    if (bucket_pred(lm->quad_tok, lm->quad_cnt, lm->slots, quad, &tok, &total, &best))
        return bucket_sample(lm->quad_tok, lm->quad_cnt, lm->slots, quad, rng, temp_shift);
    /* tri */
    if (bucket_pred(lm->tri_tok, lm->tri_cnt, lm->slots, tri, &tok, &total, &best))
        return bucket_sample(lm->tri_tok, lm->tri_cnt, lm->slots, tri, rng, temp_shift);

    /* bigram fallback: sample from bigram row weighted by count */
    {
        uint32_t x = *rng; x ^= x<<13; x ^= x>>17; x ^= x<<5; *rng = x;
        uint32_t bg_total = 0U;
        for (uint32_t i = 0; i < VOCAB_SIZE; ++i) bg_total += BG(lm, t1, i);
        if (bg_total) {
            uint32_t pick = x % bg_total;
            for (uint32_t i = 0; i < VOCAB_SIZE; ++i) {
                uint32_t c = BG(lm, t1, i);
                if (pick < c) return (uint8_t)i;
                pick -= c;
            }
        }
    }
    return (uint8_t)' ';
}

/* Emit generated text. temp_shift: 0=greedy, 1=sample(linear), 2=sample(squared) */
static void lm_emit_ex(fused_lm_t *lm, uint32_t max_len, uint32_t temp_shift, uint32_t seed) {
    char out[4096];
    uint32_t len = 0;
    uint8_t prev = 0U;
    uint8_t run = 0U;
    uint32_t rng = seed ? seed : 0xDEADBEEFU;

    while (len + 1U < sizeof(out) && len < max_len) {
        uint8_t tok;
        if (temp_shift == 0U) {
            uint8_t conf = 0;
            tok = lm_predict(lm, &conf);
        } else {
            tok = lm_sample(lm, &rng, temp_shift);
        }

        if (tok == 0U) tok = (uint8_t)' ';

        if (tok == prev) run++; else run = 0U;
        prev = tok;

        /* break degenerate repetition */
        if (run > 6U) {
            uint32_t x = rng; x^=x<<13; x^=x>>17; x^=x<<5; rng=x;
            tok = (uint8_t)(32U + x % 64U);
            run = 0U;
        }
        if (run > 10U && len > 20U) break;

        /* break looping sequences */
        if (len >= 40U) {
            uint32_t wlen = 12U;
            int looped = 0;
            for (uint32_t j = 0; j + wlen <= len - wlen; ++j) {
                if (memcmp(out + len - wlen, out + j, wlen) == 0) {
                    looped = 1; break;
                }
            }
            if (looped) break;
        }

        out[len++] = (char)tok;
        lm->lxor = lxor_upd(lm->lxor, tok);
        push_ctx(lm, tok);

        if (tok == '\n' && len > 24U) break;
        if ((tok == '.' || tok == '!' || tok == '?') && len > 60U) break;
    }

    out[len] = '\0';
    fputs(out, stdout);
}

static void lm_emit_with_guard(fused_lm_t *lm, uint32_t max_len) {
    lm_emit_ex(lm, max_len, 0U, 0U);
}

static void lm_reset_context(fused_lm_t *lm) {
    memset(lm->work, 0, sizeof(lm->work));
    lm->work_head = 0;
    lm->lxor = 0U;
}

static int read_lines_train_list(FILE *listf, fused_lm_t *lm, uint64_t *bytes_out, uint32_t *files_out) {
    char line[2048];

    while (fgets(line, sizeof(line), listf)) {
        char *cur = line;
        size_t n;
        FILE *f;
        unsigned char buf[1U << 20];
        uint64_t local_bytes = 0;

        if ((unsigned char)cur[0] == 0xEFU && (unsigned char)cur[1] == 0xBBU && (unsigned char)cur[2] == 0xBFU) {
            cur += 3;
        }

        n = strlen(cur);

        while (n > 0 && (cur[n - 1] == '\n' || cur[n - 1] == '\r' || cur[n - 1] == ' ' || cur[n - 1] == '\t')) {
            cur[--n] = '\0';
        }
        if (n == 0 || cur[0] == '#') continue;

        f = fopen(cur, "rb");
        if (!f) {
            fprintf(stderr, "warn: could not open %s\n", cur);
            continue;
        }

        lm_reset_context(lm);
        for (;;) {
            size_t got = fread(buf, 1, sizeof(buf), f);
            if (got == 0) break;
            local_bytes += (uint64_t)got;
            for (size_t i = 0; i < got; ++i) lm_update(lm, buf[i]);
        }
        fclose(f);

        *bytes_out += local_bytes;
        (*files_out)++;
    }
    return 1;
}

static int train_stream_file(fused_lm_t *lm, const char *path, uint32_t passes, uint64_t *bytes_out) {
    for (uint32_t p = 0; p < passes; ++p) {
        FILE *f = fopen(path, "rb");
        unsigned char buf[1U << 20];
        uint64_t local_bytes = 0;
        if (!f) return 0;

        lm_reset_context(lm);
        for (;;) {
            size_t got = fread(buf, 1, sizeof(buf), f);
            if (got == 0) break;
            local_bytes += (uint64_t)got;
            for (size_t i = 0; i < got; ++i) lm_update(lm, buf[i]);
        }
        fclose(f);
        *bytes_out += local_bytes;
    }
    return 1;
}

static int train_stream_list(fused_lm_t *lm, const char *list_path, uint32_t passes, uint64_t *bytes_out, uint32_t *files_out) {
    for (uint32_t p = 0; p < passes; ++p) {
        FILE *listf = fopen(list_path, "rb");
        if (!listf) return 0;
        if (!read_lines_train_list(listf, lm, bytes_out, files_out)) {
            fclose(listf);
            return 0;
        }
        fclose(listf);
    }
    return 1;
}

static int eval_stream_file(fused_lm_t *lm, const char *path, int online, uint64_t *correct_out, uint64_t *n_out) {
    FILE *f = fopen(path, "rb");
    unsigned char chunk[1U << 20];
    if (!f) return 0;

    lm_reset_context(lm);
    for (;;) {
        size_t got = fread(chunk, 1, sizeof(chunk), f);
        if (got == 0) break;
        for (size_t i = 0; i < got; ++i) {
            uint8_t conf = 0;
            uint8_t pred = lm_predict(lm, &conf);
            if (pred == chunk[i]) (*correct_out)++;
            (*n_out)++;
            if (online) lm_update(lm, chunk[i]);
            else {
                lm->lxor = lxor_upd(lm->lxor, chunk[i]);
                push_ctx(lm, chunk[i]);
            }
        }
    }
    fclose(f);
    return 1;
}

static int eval_stream_list(fused_lm_t *lm, const char *list_path, int online, uint64_t *correct_out, uint64_t *n_out, uint32_t *files_out) {
    FILE *listf = fopen(list_path, "rb");
    char line[2048];
    if (!listf) return 0;

    while (fgets(line, sizeof(line), listf)) {
        char *cur = line;
        size_t n;
        if ((unsigned char)cur[0] == 0xEFU && (unsigned char)cur[1] == 0xBBU && (unsigned char)cur[2] == 0xBFU) {
            cur += 3;
        }

        n = strlen(cur);
        while (n > 0 && (cur[n - 1] == '\n' || cur[n - 1] == '\r' || cur[n - 1] == ' ' || cur[n - 1] == '\t')) {
            cur[--n] = '\0';
        }
        if (n == 0 || cur[0] == '#') continue;

        if (!eval_stream_file(lm, cur, online, correct_out, n_out)) {
            fprintf(stderr, "warn: could not eval %s\n", cur);
            continue;
        }
        (*files_out)++;
    }

    fclose(listf);
    return 1;
}

static int write_block(FILE *f, const void *ptr, size_t nbytes) {
    return fwrite(ptr, 1, nbytes, f) == nbytes;
}

static int read_block(FILE *f, void *ptr, size_t nbytes) {
    return fread(ptr, 1, nbytes, f) == nbytes;
}

static int save_model(const fused_lm_t *lm, const char *path) {
    FILE *f = fopen(path, "wb");
    model_header_t h;
    size_t nslots;

    if (!f) return 0;

    h.magic = MAGIC;
    h.version = 1U;
    h.slots = lm->slots;
    h.ctx_count = lm->ctx_count;
    h.reserved = 0U;

    nslots = (size_t)lm->slots * (size_t)lm->ctx_count;

    if (!write_block(f, &h, sizeof(h)) ||
        !write_block(f, lm->unigram, VOCAB_SIZE * sizeof(uint32_t)) ||
        !write_block(f, lm->bigram, VOCAB_SIZE * VOCAB_SIZE * sizeof(uint16_t)) ||
        !write_block(f, lm->tri_tok, nslots * sizeof(uint8_t)) ||
        !write_block(f, lm->tri_cnt, nslots * sizeof(uint16_t)) ||
        !write_block(f, lm->quad_tok, nslots * sizeof(uint8_t)) ||
        !write_block(f, lm->quad_cnt, nslots * sizeof(uint16_t)) ||
        !write_block(f, lm->oct_tok, nslots * sizeof(uint8_t)) ||
        !write_block(f, lm->oct_cnt, nslots * sizeof(uint16_t)) ||
        !write_block(f, lm->ent_tok, nslots * sizeof(uint8_t)) ||
        !write_block(f, lm->ent_cnt, nslots * sizeof(uint16_t))) {
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

static int load_model(fused_lm_t *lm, const char *path) {
    FILE *f = fopen(path, "rb");
    model_header_t h;
    size_t nslots;

    if (!f) return 0;
    if (!read_block(f, &h, sizeof(h))) {
        fclose(f);
        return 0;
    }
    if (h.magic != MAGIC || h.version != 1U) {
        fclose(f);
        return 0;
    }
    if (!alloc_tables(lm, h.slots, h.ctx_count)) {
        fclose(f);
        return 0;
    }

    nslots = (size_t)lm->slots * (size_t)lm->ctx_count;

    if (!read_block(f, lm->unigram, VOCAB_SIZE * sizeof(uint32_t)) ||
        !read_block(f, lm->bigram, VOCAB_SIZE * VOCAB_SIZE * sizeof(uint16_t)) ||
        !read_block(f, lm->tri_tok, nslots * sizeof(uint8_t)) ||
        !read_block(f, lm->tri_cnt, nslots * sizeof(uint16_t)) ||
        !read_block(f, lm->quad_tok, nslots * sizeof(uint8_t)) ||
        !read_block(f, lm->quad_cnt, nslots * sizeof(uint16_t)) ||
        !read_block(f, lm->oct_tok, nslots * sizeof(uint8_t)) ||
        !read_block(f, lm->oct_cnt, nslots * sizeof(uint16_t)) ||
        !read_block(f, lm->ent_tok, nslots * sizeof(uint8_t)) ||
        !read_block(f, lm->ent_cnt, nslots * sizeof(uint16_t))) {
        fclose(f);
        free_tables(lm);
        return 0;
    }

    fclose(f);
    lm_reset_context(lm);
    return 1;
}

static void usage(const char *exe) {
    printf("Usage:\n");
    printf("  %s train <corpus.txt> <model.bin> [passes] [slots] [ctx_count]\n", exe);
    printf("  %s train-list <files.txt> <model.bin> [passes] [slots] [ctx_count]\n", exe);
    printf("  %s resume <model.bin> <corpus.txt> [passes]\n", exe);
    printf("  %s resume-list <model.bin> <files.txt> [passes]\n", exe);
    printf("  %s eval  <model.bin> <test.txt> [online=1]\n", exe);
    printf("  %s eval-list <model.bin> <files.txt> [online=0]\n", exe);
    printf("  %s gen   <model.bin> <prompt> [max_len=256]\n", exe);
    printf("  %s gen-sample <model.bin> <prompt> [max_len=256] [temp=1] [seed=0]\n", exe);
    printf("  %s chat  <model.bin>\n", exe);
    printf("  %s merge <base.bin> <shard.bin> [shard2.bin ...] <out.bin>\n", exe);
    printf("\nDefaults: passes=8 slots=16 ctx_count=32768\n");
    printf("\nfiles.txt format: one corpus file path per line, '#' for comments.\n");
    printf("merge: combines table counts from shard models into base model (data-parallel)\n");
    printf("gen-sample temp: 0=greedy 1=count-weighted 2=count^2-weighted (no floats)\n");
}

static int cmd_train(int argc, char **argv) {
    const char *corpus_path = argv[2];
    const char *model_path = argv[3];
    uint32_t passes = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 8U;
    uint32_t slots = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : 16U;
    uint32_t ctx_count = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : 32768U;

    fused_lm_t lm;
    uint64_t bytes = 0;
    clock_t t0, t1;

    if (!alloc_tables(&lm, slots, ctx_count)) {
        fprintf(stderr, "Failed to allocate model tables.\n");
        return 1;
    }

    lm_reset_context(&lm);
    t0 = clock();

    if (!train_stream_file(&lm, corpus_path, passes, &bytes)) {
        fprintf(stderr, "Failed to stream corpus: %s\n", corpus_path);
        free_tables(&lm);
        return 1;
    }

    t1 = clock();

    if (!save_model(&lm, model_path)) {
        fprintf(stderr, "Failed to save model: %s\n", model_path);
        free_tables(&lm);
        return 1;
    }

    printf("trained: corpus=%s bytes=%llu passes=%u slots=%u ctx=%u\n",
           corpus_path, (unsigned long long)bytes, passes, slots, ctx_count);
    printf("saved: %s\n", model_path);
    printf("train_time: %.2fs\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    free_tables(&lm);
    return 0;
}

static int cmd_train_list(int argc, char **argv) {
    const char *list_path = argv[2];
    const char *model_path = argv[3];
    uint32_t passes = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 8U;
    uint32_t slots = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : 16U;
    uint32_t ctx_count = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : 32768U;

    fused_lm_t lm;
    uint64_t bytes = 0;
    uint32_t files = 0;
    clock_t t0, t1;

    if (!alloc_tables(&lm, slots, ctx_count)) {
        fprintf(stderr, "Failed to allocate model tables.\n");
        return 1;
    }

    lm_reset_context(&lm);
    t0 = clock();

    if (!train_stream_list(&lm, list_path, passes, &bytes, &files)) {
        fprintf(stderr, "Failed to train from list: %s\n", list_path);
        free_tables(&lm);
        return 1;
    }

    t1 = clock();

    if (!save_model(&lm, model_path)) {
        fprintf(stderr, "Failed to save model: %s\n", model_path);
        free_tables(&lm);
        return 1;
    }

    printf("trained-list: list=%s files=%u bytes=%llu passes=%u slots=%u ctx=%u\n",
           list_path, files, (unsigned long long)bytes, passes, slots, ctx_count);
    printf("saved: %s\n", model_path);
    printf("train_time: %.2fs\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    free_tables(&lm);
    return 0;
}

static int cmd_resume(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *corpus_path = argv[3];
    uint32_t passes = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 1U;

    fused_lm_t lm;
    uint64_t bytes = 0;
    clock_t t0, t1;

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    t0 = clock();
    if (!train_stream_file(&lm, corpus_path, passes, &bytes)) {
        fprintf(stderr, "Failed to stream corpus: %s\n", corpus_path);
        free_tables(&lm);
        return 1;
    }
    t1 = clock();

    if (!save_model(&lm, model_path)) {
        fprintf(stderr, "Failed to update model: %s\n", model_path);
        free_tables(&lm);
        return 1;
    }

    printf("resumed: model=%s corpus=%s bytes=%llu passes=%u\n",
           model_path, corpus_path, (unsigned long long)bytes, passes);
    printf("resume_time: %.2fs\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    free_tables(&lm);
    return 0;
}

static int cmd_resume_list(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *list_path = argv[3];
    uint32_t passes = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 1U;

    fused_lm_t lm;
    uint64_t bytes = 0;
    uint32_t files = 0;
    clock_t t0, t1;

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    t0 = clock();
    if (!train_stream_list(&lm, list_path, passes, &bytes, &files)) {
        fprintf(stderr, "Failed to stream list: %s\n", list_path);
        free_tables(&lm);
        return 1;
    }
    t1 = clock();

    if (!save_model(&lm, model_path)) {
        fprintf(stderr, "Failed to update model: %s\n", model_path);
        free_tables(&lm);
        return 1;
    }

    printf("resumed-list: model=%s list=%s files=%u bytes=%llu passes=%u\n",
           model_path, list_path, files, (unsigned long long)bytes, passes);
    printf("resume_time: %.2fs\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    free_tables(&lm);
    return 0;
}

static int cmd_eval(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *test_path = argv[3];
    int online = (argc > 4) ? atoi(argv[4]) : 1;

    fused_lm_t lm;
    uint64_t correct = 0;
    uint64_t n = 0;
    clock_t t0, t1;

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    t0 = clock();

    if (!eval_stream_file(&lm, test_path, online, &correct, &n)) {
        fprintf(stderr, "Failed to load test file: %s\n", test_path);
        free_tables(&lm);
        return 1;
    }

    t1 = clock();

    {
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
        if (elapsed <= 0.0) elapsed = 1.0 / CLOCKS_PER_SEC;

        printf("eval_file: %s\n", test_path);
        printf("tokens: %llu\n", (unsigned long long)n);
        printf("accuracy: %.2f%%\n", n ? (100.0 * (double)correct / (double)n) : 0.0);
        printf("mode: %s\n", online ? "online-update" : "frozen-weights");
        printf("speed: %.2f Mtok/s\n", n ? ((double)n / elapsed) / 1e6 : 0.0);
    }

    free_tables(&lm);
    return 0;
}

static int cmd_eval_list(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *list_path = argv[3];
    int online = (argc > 4) ? atoi(argv[4]) : 0;

    fused_lm_t lm;
    uint64_t correct = 0;
    uint64_t n = 0;
    uint32_t files = 0;
    clock_t t0, t1;

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    t0 = clock();
    if (!eval_stream_list(&lm, list_path, online, &correct, &n, &files)) {
        fprintf(stderr, "Failed to load eval list: %s\n", list_path);
        free_tables(&lm);
        return 1;
    }
    t1 = clock();

    {
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
        if (elapsed <= 0.0) elapsed = 1.0 / CLOCKS_PER_SEC;

        printf("eval_list: %s\n", list_path);
        printf("files: %u\n", files);
        printf("tokens: %llu\n", (unsigned long long)n);
        printf("accuracy: %.2f%%\n", n ? (100.0 * (double)correct / (double)n) : 0.0);
        printf("mode: %s\n", online ? "online-update" : "frozen-weights");
        printf("speed: %.2f Mtok/s\n", n ? ((double)n / elapsed) / 1e6 : 0.0);
    }

    free_tables(&lm);
    return 0;
}

static int cmd_gen(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *prompt = argv[3];
    uint32_t max_len = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 256U;

    fused_lm_t lm;

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    lm_reset_context(&lm);
    lm_prime_ctx(&lm, (const uint8_t *)prompt, (uint32_t)strlen(prompt));

    printf("%s", prompt);
    lm_emit_with_guard(&lm, max_len);
    putchar('\n');

    free_tables(&lm);
    return 0;
}

/* gen-sample <model> <prompt> [max_len=256] [temp=1] [seed=0]
 * temp: 0=greedy, 1=sample proportional to count, 2=sample proportional to count^2 */
static int cmd_gen_sample(int argc, char **argv) {
    const char *model_path = argv[2];
    const char *prompt     = argv[3];
    uint32_t max_len   = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 256U;
    uint32_t temp      = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : 1U;
    uint32_t seed      = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : (uint32_t)time(NULL);

    if (temp > 4U) temp = 4U; /* cap to avoid overflow */

    fused_lm_t lm;

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    lm_reset_context(&lm);
    lm_prime_ctx(&lm, (const uint8_t *)prompt, (uint32_t)strlen(prompt));

    printf("%s", prompt);
    lm_emit_ex(&lm, max_len, temp, seed);
    putchar('\n');

    free_tables(&lm);
    return 0;
}

static int cmd_chat(int argc, char **argv) {
    const char *model_path = argv[2];
    fused_lm_t lm;
    char line[512];

    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    (void)argc;

    printf("chat mode (type :quit to exit, :reset to clear context)\n");
    lm_reset_context(&lm);

    for (;;) {
        printf("you> ");
        if (!fgets(line, sizeof(line), stdin)) break;

        if (strncmp(line, ":quit", 5) == 0) break;
        if (strncmp(line, ":reset", 6) == 0) {
            lm_reset_context(&lm);
            printf("assistant> context reset\n");
            continue;
        }

        lm_prime_ctx(&lm, (const uint8_t *)"USER: ", 6U);
        lm_prime_ctx(&lm, (const uint8_t *)line, (uint32_t)strlen(line));
        lm_prime_ctx(&lm, (const uint8_t *)"ASSISTANT: ", 11U);

        printf("assistant> ");
        lm_emit_with_guard(&lm, 300U);
        putchar('\n');
    }

    free_tables(&lm);
    return 0;
}

/* ── model merge: add shard model table counts into base (data-parallel) ─── */
static int merge_into(fused_lm_t *base, fused_lm_t *shard) {
    if (base->slots != shard->slots || base->ctx_count != shard->ctx_count) {
        fprintf(stderr, "merge: geometry mismatch (slots/ctx_count must match)\n");
        return 0;
    }
    uint32_t n = base->slots * base->ctx_count;

    /* unigram */
    for (uint32_t i = 0; i < 256U; i++) {
        uint64_t s = (uint64_t)base->unigram[i] + shard->unigram[i];
        base->unigram[i] = s > 0xFFFFFFFFUL ? 0xFFFFFFFFU : (uint32_t)s;
    }
    /* bigram */
    for (uint32_t i = 0; i < 256U*256U; i++)
        base->bigram[i] = sat16(base->bigram[i], shard->bigram[i]);
    /* tri */
    for (uint32_t i = 0; i < n; i++)
        base->tri_cnt[i] = sat16(base->tri_cnt[i], shard->tri_cnt[i]);
    /* quad */
    for (uint32_t i = 0; i < n; i++)
        base->quad_cnt[i] = sat16(base->quad_cnt[i], shard->quad_cnt[i]);
    /* oct */
    for (uint32_t i = 0; i < n; i++)
        base->oct_cnt[i] = sat16(base->oct_cnt[i], shard->oct_cnt[i]);
    /* ent */
    for (uint32_t i = 0; i < n; i++)
        base->ent_cnt[i] = sat16(base->ent_cnt[i], shard->ent_cnt[i]);

    return 1;
}

static int cmd_merge(int argc, char **argv) {
    /* argv: merge <base.bin> <shard1.bin> [shard2...] <out.bin> */
    /* argc >= 4: [0]=exe [1]=merge [2]=base [3..argc-2]=shards [argc-1]=out */
    if (argc < 4) { usage(argv[0]); return 1; }

    const char *base_path = argv[2];
    const char *out_path  = argv[argc - 1];
    int n_shards = argc - 4; /* files between base and out */

    fused_lm_t base;
    if (!load_model(&base, base_path)) {
        fprintf(stderr, "merge: cannot load base model: %s\n", base_path);
        return 1;
    }

    printf("[merge] base: %s  shards: %d  out: %s\n", base_path, n_shards, out_path);

    for (int s = 0; s < n_shards; s++) {
        const char *shard_path = argv[3 + s];
        fused_lm_t shard;
        if (!load_model(&shard, shard_path)) {
            fprintf(stderr, "merge: cannot load shard: %s\n", shard_path);
            free_tables(&base);
            return 1;
        }
        printf("  + merging shard %d: %s\n", s + 1, shard_path);
        if (!merge_into(&base, &shard)) {
            free_tables(&shard);
            free_tables(&base);
            return 1;
        }
        free_tables(&shard);
    }

    if (!save_model(&base, out_path)) {
        fprintf(stderr, "merge: cannot save output: %s\n", out_path);
        free_tables(&base);
        return 1;
    }
    printf("[merge] saved merged model -> %s\n", out_path);
    free_tables(&base);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_train(argc, argv);
    }
    if (strcmp(argv[1], "train-list") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_train_list(argc, argv);
    }
    if (strcmp(argv[1], "resume") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_resume(argc, argv);
    }
    if (strcmp(argv[1], "resume-list") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_resume_list(argc, argv);
    }
    if (strcmp(argv[1], "eval") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_eval(argc, argv);
    }
    if (strcmp(argv[1], "eval-list") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_eval_list(argc, argv);
    }
    if (strcmp(argv[1], "gen") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_gen(argc, argv);
    }
    if (strcmp(argv[1], "gen-sample") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_gen_sample(argc, argv);
    }
    if (strcmp(argv[1], "chat") == 0) {
        if (argc < 3) { usage(argv[0]); return 1; }
        return cmd_chat(argc, argv);
    }
    if (strcmp(argv[1], "merge") == 0) {
        /* merge <base.bin> <shard1.bin> [shard2.bin ...] <out.bin> */
        if (argc < 4) { usage(argv[0]); return 1; }
        return cmd_merge(argc, argv);
    }

    usage(argv[0]);
    return 1;
}
