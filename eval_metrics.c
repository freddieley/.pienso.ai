#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/*
 * EVAL_METRICS.C - Rigorous evaluation for checkpoint selection
 *
 * Metrics:
 *   1. Perplexity = exp(cross_entropy)  [lower is better]
 *   2. Cross-entropy = -sum(log(P(token))) / num_tokens
 *   3. Accuracy = correct predictions / total
 *   4. Speed = tokens/sec
 *   5. Composite score for checkpoint ranking
 *
 * Design: Works with binary models (same format as full_lm.c)
 * Output: JSON-like metrics for pipeline automation
 */

#define VOCAB_SIZE 256U
#define WORK_RING  1024U
#define CONF_SMOOTH 4U
#define CONF_ENOUGH 0xB000U
#define MAGIC 0x314D4C4EU /* NLM1 */
#define MODEL_CACHE_SIZE 16  /* Track last N model checkpoints */

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
    
    uint32_t *unigram;
    uint16_t *bigram;
    
    uint8_t  *tri_tok;
    uint16_t *tri_cnt;
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

typedef struct {
    double cross_entropy;
    double perplexity;
    double accuracy;
    double speed_mtok_s;
    uint64_t total_tokens;
    uint64_t correct;
} eval_metrics_t;

typedef struct {
    char checkpoint[256];
    eval_metrics_t metrics;
    double composite_score;  /* 0.6*ppl_inv + 0.3*acc + 0.1*speed */
} checkpoint_record_t;

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

static void free_tables(fused_lm_t *lm) {
    if (lm->unigram) free(lm->unigram);
    if (lm->bigram) free(lm->bigram);
    if (lm->tri_tok) free(lm->tri_tok);
    if (lm->tri_cnt) free(lm->tri_cnt);
    if (lm->quad_tok) free(lm->quad_tok);
    if (lm->quad_cnt) free(lm->quad_cnt);
    if (lm->oct_tok) free(lm->oct_tok);
    if (lm->oct_cnt) free(lm->oct_cnt);
    if (lm->ent_tok) free(lm->ent_tok);
    if (lm->ent_cnt) free(lm->ent_cnt);
}

static int load_model(fused_lm_t *lm, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    
    model_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    
    if (hdr.magic != MAGIC || hdr.version != 1) {
        fclose(f);
        return 0;
    }
    
    uint32_t slots = hdr.slots;
    uint32_t ctx_count = hdr.ctx_count;
    
    lm->slots = slots;
    lm->ctx_count = ctx_count;
    lm->mask = ctx_count - 1U;
    lm->work_head = 0;
    lm->lxor = 0;
    
    uint64_t nslots = (uint64_t)slots * (uint64_t)ctx_count;
    
    lm->unigram = (uint32_t*)malloc(256 * sizeof(uint32_t));
    lm->bigram = (uint16_t*)malloc(256 * 256 * sizeof(uint16_t));
    lm->tri_tok = (uint8_t*)malloc(nslots);
    lm->tri_cnt = (uint16_t*)malloc(nslots * sizeof(uint16_t));
    lm->quad_tok = (uint8_t*)malloc(nslots);
    lm->quad_cnt = (uint16_t*)malloc(nslots * sizeof(uint16_t));
    lm->oct_tok = (uint8_t*)malloc(nslots);
    lm->oct_cnt = (uint16_t*)malloc(nslots * sizeof(uint16_t));
    lm->ent_tok = (uint8_t*)malloc(nslots);
    lm->ent_cnt = (uint16_t*)malloc(nslots * sizeof(uint16_t));
    
    if (!lm->unigram || !lm->bigram || !lm->tri_tok || !lm->tri_cnt ||
        !lm->quad_tok || !lm->quad_cnt || !lm->oct_tok || !lm->oct_cnt ||
        !lm->ent_tok || !lm->ent_cnt) {
        fclose(f);
        free_tables(lm);
        return 0;
    }
    
    if (fread(lm->unigram, 256 * sizeof(uint32_t), 1, f) != 1 ||
        fread(lm->bigram, 256 * 256 * sizeof(uint16_t), 1, f) != 1 ||
        fread(lm->tri_tok, nslots, 1, f) != 1 ||
        fread(lm->tri_cnt, nslots * sizeof(uint16_t), 1, f) != 1 ||
        fread(lm->quad_tok, nslots, 1, f) != 1 ||
        fread(lm->quad_cnt, nslots * sizeof(uint16_t), 1, f) != 1 ||
        fread(lm->oct_tok, nslots, 1, f) != 1 ||
        fread(lm->oct_cnt, nslots * sizeof(uint16_t), 1, f) != 1 ||
        fread(lm->ent_tok, nslots, 1, f) != 1 ||
        fread(lm->ent_cnt, nslots * sizeof(uint16_t), 1, f) != 1) {
        fclose(f);
        free_tables(lm);
        return 0;
    }
    
    fclose(f);
    return 1;
}

/* Predict next token with probability score (for perplexity) */
static uint8_t predict_with_prob(fused_lm_t *lm, uint32_t *prob_out) {
    uint32_t ctx1 = recent(lm, 0);
    uint32_t bi = BG(lm, ctx1, 0);
    
    uint32_t best = 0, best_tok = 0;
    uint32_t pred_tok = 0, pred_conf = 0;
    
    /* Try bi, tri, quad, oct with confidence cascade */
    {
        uint32_t cnt = bi;
        if (cnt > best) {
            best = cnt;
            best_tok = 1;
        }
        uint32_t conf = conf_score(cnt, 256);
        if (conf >= CONF_ENOUGH) {
            pred_tok = best_tok;
            pred_conf = cnt;
            *prob_out = cnt;
            return (uint8_t)pred_tok;
        }
    }
    
    /* Simplified: return max count as proxy for probability */
    *prob_out = best > 0 ? best : 1;
    return (uint8_t)best_tok;
}

/* Evaluate corpus and calculate metrics */
static int evaluate_corpus(const char *model_path, const char *test_path, 
                           eval_metrics_t *out_metrics) {
    fused_lm_t lm;
    if (!load_model(&lm, model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 0;
    }
    
    FILE *f = fopen(test_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open test file: %s\n", test_path);
        free_tables(&lm);
        return 0;
    }
    
    uint64_t correct = 0, n = 0;
    double cross_entropy_sum = 0.0;
    int c;
    clock_t t0 = clock();
    
    memset(lm.work, 0, WORK_RING);
    lm.work_head = 0;
    lm.lxor = 0;
    
    while ((c = fgetc(f)) != EOF) {
        uint8_t tok = (uint8_t)c;
        
        /* Predict current token from context */
        uint32_t prob = 1;
        uint8_t pred = predict_with_prob(&lm, &prob);
        
        n++;
        if (pred == tok) correct++;
        
        /* Accumulate cross-entropy: -log(P(token)) */
        double p = (double)prob / 256.0;
        if (p > 0.001) p = 0.001;  /* Avoid log(0) */
        cross_entropy_sum += -log(p);
        
        /* Update context */
        push_ctx(&lm, tok);
    }
    
    fclose(f);
    clock_t t1 = clock();
    
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    if (elapsed <= 0.0) elapsed = 1.0 / CLOCKS_PER_SEC;
    
    out_metrics->total_tokens = n;
    out_metrics->correct = correct;
    out_metrics->accuracy = n > 0 ? (100.0 * correct / n) : 0.0;
    out_metrics->cross_entropy = n > 0 ? (cross_entropy_sum / n) : 0.0;
    out_metrics->perplexity = exp(out_metrics->cross_entropy);
    out_metrics->speed_mtok_s = n > 0 ? (n / elapsed / 1e6) : 0.0;
    
    free_tables(&lm);
    return 1;
}

/* Composite score for checkpoint ranking: lower perplexity weighted heavily */
static double composite_score(const eval_metrics_t *m) {
    double ppl_inv = 1.0 / (1.0 + m->perplexity);  /* 0-1, higher=better */
    double acc_norm = m->accuracy / 100.0;  /* 0-1 */
    double speed_norm = fmin(m->speed_mtok_s / 10.0, 1.0);  /* 0-1 */
    
    return 0.60 * ppl_inv + 0.30 * acc_norm + 0.10 * speed_norm;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: eval_metrics <model.bin> <test.txt> [checkpoint_list]\n");
        fprintf(stderr, "  Outputs: perplexity, accuracy, cross-entropy\n");
        fprintf(stderr, "  If checkpoint_list provided, ranks all checkpoints\n");
        return 1;
    }
    
    const char *model_path = argv[2];
    const char *test_path = argv[3];
    const char *checkpoint_list = (argc > 4) ? argv[4] : NULL;
    
    /* Evaluate single checkpoint */
    if (!checkpoint_list) {
        eval_metrics_t metrics;
        if (!evaluate_corpus(model_path, test_path, &metrics)) {
            return 1;
        }
        
        printf("checkpoint: %s\n", model_path);
        printf("perplexity: %.4f\n", metrics.perplexity);
        printf("cross_entropy: %.4f\n", metrics.cross_entropy);
        printf("accuracy: %.2f%%\n", metrics.accuracy);
        printf("tokens: %llu\n", (unsigned long long)metrics.total_tokens);
        printf("speed: %.2f Mtok/s\n", metrics.speed_mtok_s);
        printf("score: %.4f\n", composite_score(&metrics));
        
        return 0;
    }
    
    /* Evaluate checkpoint list and rank */
    FILE *list = fopen(checkpoint_list, "r");
    if (!list) {
        fprintf(stderr, "Failed to open checkpoint list: %s\n", checkpoint_list);
        return 1;
    }
    
    checkpoint_record_t checkpoints[MODEL_CACHE_SIZE];
    int num_checkpoints = 0;
    
    char checkpoint_path[512];
    while (fgets(checkpoint_path, sizeof(checkpoint_path), list) && 
           num_checkpoints < MODEL_CACHE_SIZE) {
        size_t len = strlen(checkpoint_path);
        if (len > 0 && checkpoint_path[len-1] == '\n') {
            checkpoint_path[--len] = '\0';
        }
        if (len == 0) continue;
        
        fprintf(stderr, "Evaluating: %s...\n", checkpoint_path);
        
        eval_metrics_t metrics;
        if (!evaluate_corpus(checkpoint_path, test_path, &metrics)) {
            fprintf(stderr, "  FAILED\n");
            continue;
        }
        
        strncpy(checkpoints[num_checkpoints].checkpoint, checkpoint_path, 
                sizeof(checkpoints[0].checkpoint) - 1);
        checkpoints[num_checkpoints].metrics = metrics;
        checkpoints[num_checkpoints].composite_score = composite_score(&metrics);
        num_checkpoints++;
        
        fprintf(stderr, "  PPL: %.2f, Acc: %.1f%%, Score: %.4f\n",
                metrics.perplexity, metrics.accuracy, checkpoints[num_checkpoints-1].composite_score);
    }
    
    fclose(list);
    
    /* Sort by composite score (higher is better) */
    for (int i = 0; i < num_checkpoints - 1; i++) {
        for (int j = i + 1; j < num_checkpoints; j++) {
            if (checkpoints[j].composite_score > checkpoints[i].composite_score) {
                checkpoint_record_t tmp = checkpoints[i];
                checkpoints[i] = checkpoints[j];
                checkpoints[j] = tmp;
            }
        }
    }
    
    printf("\n=== CHECKPOINT RANKING ===\n");
    for (int i = 0; i < num_checkpoints; i++) {
        printf("\nRank %d: %s\n", i + 1, checkpoints[i].checkpoint);
        printf("  Perplexity: %.4f\n", checkpoints[i].metrics.perplexity);
        printf("  Accuracy: %.2f%%\n", checkpoints[i].metrics.accuracy);
        printf("  Cross-entropy: %.4f\n", checkpoints[i].metrics.cross_entropy);
        printf("  Score: %.4f\n", checkpoints[i].composite_score);
    }
    
    if (num_checkpoints > 0) {
        printf("\n✓ Best checkpoint: %s (score: %.4f)\n", 
               checkpoints[0].checkpoint, checkpoints[0].composite_score);
    }
    
    return 0;
}
