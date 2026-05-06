#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOCAB_SIZE 256
#define WORKING_RING 1024

#define MAX_MODELS 5
#define MAX_CONFIGS 4
#define MAX_NAME 48

typedef enum {
    MODEL_BIGRAM = 0,
    MODEL_TRIGRAM = 1,
    MODEL_FOURGRAM = 2,
    MODEL_OCTAGRAM = 3,
    MODEL_CASCADE = 4
} model_kind_t;

typedef struct {
    uint8_t token;
    uint16_t count;
} slot_t;

typedef struct {
    slot_t *slot;
} bucket_t;

typedef struct {
    char name[MAX_NAME];
    model_kind_t kind;
    uint32_t slots;
    uint32_t ctx_count;
    uint32_t bootstrap_passes;
    uint32_t test_steps;
} run_config_t;

typedef struct {
    model_kind_t kind;
    uint32_t slots;
    uint32_t ctx_count;

    uint32_t *unigram;
    uint16_t *bigram;
    bucket_t *trigram;
    bucket_t *fourgram;
    bucket_t *octagram;

    uint8_t working[WORKING_RING];
    uint32_t working_head;

    uint64_t seen;
    uint64_t correct;
} runtime_t;

typedef struct {
    char config_name[MAX_NAME];
    char model_name[MAX_NAME];
    double accuracy;
    double tokens_per_sec;
    double memory_mb;
    double score;
} bench_result_t;

static inline uint16_t sat_u16_add(uint16_t v, uint16_t add) {
    uint32_t n = (uint32_t)v + (uint32_t)add;
    return (uint16_t)(n > 65535U ? 65535U : n);
}

static inline uint8_t recent_tok(const runtime_t *rt, uint32_t back) {
    uint32_t idx = (rt->working_head + WORKING_RING - 1U - back) % WORKING_RING;
    return rt->working[idx];
}

static inline void push_tok(runtime_t *rt, uint8_t tok) {
    rt->working[rt->working_head] = tok;
    rt->working_head = (rt->working_head + 1U) % WORKING_RING;
}

/* Binary-only context hash mixers (xor/shift/or), no multipliers. */
static inline uint32_t mix16(uint16_t x, uint32_t mask) {
    uint32_t h = (uint32_t)x;
    h ^= (h << 7);
    h ^= (h >> 3);
    h ^= (h << 11);
    return h & mask;
}

static inline uint32_t tri_ctx(uint8_t t2, uint8_t t1, uint32_t mask) {
    return mix16((uint16_t)(((uint16_t)t2 << 8) | (uint16_t)t1), mask);
}

static inline uint32_t quad_ctx(uint8_t t3, uint8_t t2, uint8_t t1, uint32_t mask) {
    uint32_t h = ((uint32_t)t3 << 16) | ((uint32_t)t2 << 8) | (uint32_t)t1;
    h ^= (h << 5);
    h ^= (h >> 7);
    h ^= (h << 9);
    return h & mask;
}

static inline uint32_t oct_ctx(uint8_t t7, uint8_t t6, uint8_t t5, uint8_t t4,
                               uint8_t t3, uint8_t t2, uint8_t t1, uint32_t mask) {
    uint32_t h = 0xA55AA55AU;
    h ^= ((uint32_t)t7 << 24) | ((uint32_t)t6 << 16) | ((uint32_t)t5 << 8) | (uint32_t)t4;
    h ^= ((uint32_t)t3 << 24) | ((uint32_t)t2 << 16) | ((uint32_t)t1 << 8);
    h ^= (h << 5);
    h ^= (h >> 11);
    h ^= (h << 13);
    return h & mask;
}

static int is_pow2(uint32_t v) {
    return v != 0U && (v & (v - 1U)) == 0U;
}

static int runtime_init(runtime_t *rt, model_kind_t kind, uint32_t slots, uint32_t ctx_count) {
    memset(rt, 0, sizeof(*rt));
    if (!is_pow2(ctx_count)) return 0;

    rt->kind = kind;
    rt->slots = slots;
    rt->ctx_count = ctx_count;

    rt->unigram = (uint32_t *)calloc(VOCAB_SIZE, sizeof(uint32_t));
    rt->bigram = (uint16_t *)calloc((size_t)VOCAB_SIZE * (size_t)VOCAB_SIZE, sizeof(uint16_t));
    if (!rt->unigram || !rt->bigram) return 0;

    if (kind >= MODEL_TRIGRAM) {
        rt->trigram = (bucket_t *)calloc(ctx_count, sizeof(bucket_t));
        if (!rt->trigram) return 0;
        for (uint32_t i = 0; i < ctx_count; ++i) {
            rt->trigram[i].slot = (slot_t *)calloc(slots, sizeof(slot_t));
            if (!rt->trigram[i].slot) return 0;
        }
    }
    if (kind >= MODEL_FOURGRAM) {
        rt->fourgram = (bucket_t *)calloc(ctx_count, sizeof(bucket_t));
        if (!rt->fourgram) return 0;
        for (uint32_t i = 0; i < ctx_count; ++i) {
            rt->fourgram[i].slot = (slot_t *)calloc(slots, sizeof(slot_t));
            if (!rt->fourgram[i].slot) return 0;
        }
    }
    if (kind >= MODEL_OCTAGRAM || kind == MODEL_CASCADE) {
        rt->octagram = (bucket_t *)calloc(ctx_count, sizeof(bucket_t));
        if (!rt->octagram) return 0;
        for (uint32_t i = 0; i < ctx_count; ++i) {
            rt->octagram[i].slot = (slot_t *)calloc(slots, sizeof(slot_t));
            if (!rt->octagram[i].slot) return 0;
        }
    }

    return 1;
}

static void free_buckets(bucket_t *b, uint32_t ctx_count) {
    if (!b) return;
    for (uint32_t i = 0; i < ctx_count; ++i) free(b[i].slot);
    free(b);
}

static void runtime_free(runtime_t *rt) {
    free(rt->unigram);
    free(rt->bigram);
    free_buckets(rt->trigram, rt->ctx_count);
    free_buckets(rt->fourgram, rt->ctx_count);
    free_buckets(rt->octagram, rt->ctx_count);
    memset(rt, 0, sizeof(*rt));
}

static inline uint16_t *bigram_cell(runtime_t *rt, uint8_t a, uint8_t b) {
    return &rt->bigram[(size_t)a * VOCAB_SIZE + (size_t)b];
}

static void bucket_inc(bucket_t *b, uint32_t slots, uint8_t tok) {
    int free_i = -1;
    int min_i = 0;
    uint16_t min_c = b->slot[0].count;

    for (uint32_t i = 0; i < slots; ++i) {
        if (b->slot[i].count == 0U && free_i < 0) free_i = (int)i;
        if (b->slot[i].count > 0U && b->slot[i].token == tok) {
            b->slot[i].count = sat_u16_add(b->slot[i].count, 1U);
            return;
        }
        if (b->slot[i].count < min_c) {
            min_c = b->slot[i].count;
            min_i = (int)i;
        }
    }

    if (free_i >= 0) {
        b->slot[free_i].token = tok;
        b->slot[free_i].count = 1U;
        return;
    }

    if (b->slot[min_i].count > 1U) b->slot[min_i].count--;
    else {
        b->slot[min_i].token = tok;
        b->slot[min_i].count = 1U;
    }
}

static int bucket_predict(const bucket_t *b, uint32_t slots, uint8_t *tok_out, uint8_t *conf_out) {
    uint32_t total = 0;
    uint32_t best = 0;
    uint8_t btok = 32U;

    for (uint32_t i = 0; i < slots; ++i) {
        uint16_t c = b->slot[i].count;
        total += c;
        if (c > best) {
            best = c;
            btok = b->slot[i].token;
        }
    }

    if (total == 0U) return 0;
    *tok_out = btok;
    *conf_out = (uint8_t)((best * 255U) / total);
    return 1;
}

static void update_language(runtime_t *rt, uint8_t obs) {
    uint8_t t1 = recent_tok(rt, 0);
    uint8_t t2 = recent_tok(rt, 1);
    uint8_t t3 = recent_tok(rt, 2);
    uint8_t t4 = recent_tok(rt, 3);
    uint8_t t5 = recent_tok(rt, 4);
    uint8_t t6 = recent_tok(rt, 5);
    uint8_t t7 = recent_tok(rt, 6);
    uint32_t mask = rt->ctx_count - 1U;

    rt->unigram[obs]++;
    *bigram_cell(rt, t1, obs) = sat_u16_add(*bigram_cell(rt, t1, obs), 1U);

    if (rt->kind >= MODEL_TRIGRAM || rt->kind == MODEL_CASCADE) {
        bucket_inc(&rt->trigram[tri_ctx(t2, t1, mask)], rt->slots, obs);
    }
    if (rt->kind >= MODEL_FOURGRAM || rt->kind == MODEL_CASCADE) {
        bucket_inc(&rt->fourgram[quad_ctx(t3, t2, t1, mask)], rt->slots, obs);
    }
    if (rt->kind >= MODEL_OCTAGRAM || rt->kind == MODEL_CASCADE) {
        bucket_inc(&rt->octagram[oct_ctx(t7, t6, t5, t4, t3, t2, t1, mask)], rt->slots, obs);
    }
}

static uint8_t predict(runtime_t *rt, uint8_t *conf_out) {
    uint8_t t1 = recent_tok(rt, 0);
    uint8_t t2 = recent_tok(rt, 1);
    uint8_t t3 = recent_tok(rt, 2);
    uint8_t t4 = recent_tok(rt, 3);
    uint8_t t5 = recent_tok(rt, 4);
    uint8_t t6 = recent_tok(rt, 5);
    uint8_t t7 = recent_tok(rt, 6);
    uint32_t mask = rt->ctx_count - 1U;

    uint8_t tok = 32U;
    uint8_t conf = 0;

    if (rt->kind == MODEL_CASCADE || rt->kind == MODEL_OCTAGRAM) {
        if (bucket_predict(&rt->octagram[oct_ctx(t7, t6, t5, t4, t3, t2, t1, mask)], rt->slots, &tok, &conf)) {
            *conf_out = conf;
            return tok;
        }
    }

    if (rt->kind == MODEL_CASCADE || rt->kind == MODEL_FOURGRAM) {
        if (bucket_predict(&rt->fourgram[quad_ctx(t3, t2, t1, mask)], rt->slots, &tok, &conf)) {
            *conf_out = conf;
            return tok;
        }
    }

    if (rt->kind == MODEL_CASCADE || rt->kind == MODEL_TRIGRAM) {
        if (bucket_predict(&rt->trigram[tri_ctx(t2, t1, mask)], rt->slots, &tok, &conf)) {
            *conf_out = conf;
            return tok;
        }
    }

    uint16_t best = 0;
    uint8_t btok = 32U;
    uint32_t total = 0;
    for (uint32_t i = 0; i < VOCAB_SIZE; ++i) {
        uint16_t c = *bigram_cell(rt, t1, (uint8_t)i);
        total += c;
        if (c > best) {
            best = c;
            btok = (uint8_t)i;
        }
    }

    *conf_out = total ? (uint8_t)((best * 255U) / total) : 0U;
    return btok;
}

static const unsigned char *generated_corpus(void) {
    return (const unsigned char *)
        "the system learns from direct experience over continuous time. "
        "stable online learning demands clean external feedback loops only. "
        "binary operations can build rich behavior with compact state. "
        "prediction quality improves when context is specific and fresh. "
        "resource limits force elegant design and explicit tradeoffs. ";
}

static const unsigned char *conversation_corpus(void) {
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
        "ASSISTANT: Yes, by preserving simple rules and reusable memory interfaces.\n";
}

static unsigned char *load_optional_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long sz;
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    buf = (unsigned char *)malloc((size_t)sz + 1U);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1U, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);

    buf[sz] = '\0';
    *len_out = (size_t)sz;
    return buf;
}

static double estimate_mem_mb(const run_config_t *cfg, model_kind_t kind) {
    double bytes = 0.0;
    bytes += VOCAB_SIZE * sizeof(uint32_t);
    bytes += (double)VOCAB_SIZE * VOCAB_SIZE * sizeof(uint16_t);

    if (kind >= MODEL_TRIGRAM || kind == MODEL_CASCADE)
        bytes += (double)cfg->ctx_count * cfg->slots * sizeof(slot_t);
    if (kind >= MODEL_FOURGRAM || kind == MODEL_CASCADE)
        bytes += (double)cfg->ctx_count * cfg->slots * sizeof(slot_t);
    if (kind >= MODEL_OCTAGRAM || kind == MODEL_CASCADE)
        bytes += (double)cfg->ctx_count * cfg->slots * sizeof(slot_t);

    bytes += WORKING_RING;
    return bytes / (1024.0 * 1024.0);
}

static const char *model_name(model_kind_t kind) {
    switch (kind) {
        case MODEL_BIGRAM: return "bigram";
        case MODEL_TRIGRAM: return "trigram";
        case MODEL_FOURGRAM: return "fourgram";
        case MODEL_OCTAGRAM: return "octagram";
        case MODEL_CASCADE: return "cascade";
        default: return "unknown";
    }
}

static bench_result_t run_benchmark(const run_config_t *cfg, model_kind_t kind,
                                    const unsigned char *train_stream, uint32_t train_len,
                                    const unsigned char *test_stream, uint32_t test_len) {
    bench_result_t r;
    runtime_t rt;
    clock_t t0, t1;
    uint32_t i;

    memset(&r, 0, sizeof(r));
    snprintf(r.config_name, sizeof(r.config_name), "%s", cfg->name);
    snprintf(r.model_name, sizeof(r.model_name), "%s", model_name(kind));

    if (!runtime_init(&rt, kind, cfg->slots, cfg->ctx_count)) {
        fprintf(stderr, "init failed for %s/%s\n", cfg->name, model_name(kind));
        return r;
    }

    for (uint32_t p = 0; p < cfg->bootstrap_passes; ++p) {
        for (i = 0; i < train_len; ++i) {
            uint8_t c = train_stream[i];
            update_language(&rt, c);
            push_tok(&rt, c);
        }
    }

    t0 = clock();
    {
        uint64_t executed = 0ULL;
        uint32_t batch = cfg->test_steps;
        if (batch < 20000U) batch = 20000U;

        do {
            for (i = 0; i < batch; ++i) {
                uint8_t conf = 0;
                uint8_t pred = predict(&rt, &conf);
                uint8_t obs = test_stream[(uint32_t)(executed % test_len)];
                (void)conf;

                rt.seen++;
                rt.correct += (pred == obs) ? 1U : 0U;

                update_language(&rt, obs);
                push_tok(&rt, obs);
                executed++;
            }
            t1 = clock();
        } while (((double)(t1 - t0) / (double)CLOCKS_PER_SEC) < 0.20);

        r.accuracy = rt.seen ? (double)rt.correct / (double)rt.seen : 0.0;
        {
            double dt = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
            if (dt <= 0.0) dt = 1e-9;
            r.tokens_per_sec = (double)executed / dt;
        }
    }
    r.memory_mb = estimate_mem_mb(cfg, kind);

    runtime_free(&rt);
    return r;
}

static int cmp_score_desc(const void *a, const void *b) {
    const bench_result_t *ra = (const bench_result_t *)a;
    const bench_result_t *rb = (const bench_result_t *)b;
    return (rb->score > ra->score) - (rb->score < ra->score);
}

static void score_results(bench_result_t *arr, uint32_t n) {
    double max_tps = 0.0;
    double min_mem = 1e18;
    for (uint32_t i = 0; i < n; ++i) {
        if (arr[i].tokens_per_sec > max_tps) max_tps = arr[i].tokens_per_sec;
        if (arr[i].memory_mb < min_mem) min_mem = arr[i].memory_mb;
    }
    if (max_tps <= 0.0) max_tps = 1.0;
    if (min_mem <= 0.0) min_mem = 1e-6;

    for (uint32_t i = 0; i < n; ++i) {
        double acc = arr[i].accuracy;
        double speed = arr[i].tokens_per_sec / max_tps;
        double mem_eff = min_mem / arr[i].memory_mb;
        if (mem_eff > 1.0) mem_eff = 1.0;

        arr[i].score = 0.65 * acc + 0.25 * speed + 0.10 * mem_eff;
    }
}

static void print_results(bench_result_t *arr, uint32_t n) {
    qsort(arr, n, sizeof(arr[0]), cmp_score_desc);

    printf("\n=== Binary-Only Stress Test Results ===\n");
    printf("rank | score  | acc%%   | tok/s     | mem(MB) | config/model\n");
    printf("-----+--------+--------+-----------+---------+------------------------------\n");
    for (uint32_t i = 0; i < n; ++i) {
        printf("%4u | %0.4f | %6.2f | %9.0f | %7.2f | %s/%s\n",
               i + 1U,
               arr[i].score,
               arr[i].accuracy * 100.0,
               arr[i].tokens_per_sec,
               arr[i].memory_mb,
               arr[i].config_name,
               arr[i].model_name);
    }
}

static void prime_from_text(runtime_t *rt, const unsigned char *text) {
    for (uint32_t i = 0; text[i] != '\0'; ++i) {
        uint8_t c = text[i];
        update_language(rt, c);
        push_tok(rt, c);
    }
}

static void generate_reply(runtime_t *rt, const char *prompt, char *out, uint32_t out_cap) {
    uint32_t len = 0;
    uint8_t conf = 0;
    uint8_t prev = 0;
    uint32_t streak = 0;

    prime_from_text(rt, (const unsigned char *)"USER: ");
    prime_from_text(rt, (const unsigned char *)prompt);
    prime_from_text(rt, (const unsigned char *)"\nASSISTANT: ");

    while (len + 1U < out_cap) {
        uint8_t tok = predict(rt, &conf);
        if (tok == '\0') tok = ' ';

        if (tok == prev) streak++;
        else streak = 0;
        prev = tok;

        if (streak > 8U) tok = ' ';
        if (streak > 14U && len > 32U) break;

        out[len++] = (char)tok;

        /* Keep generation context flowing without learning from own output. */
        push_tok(rt, tok);

        if (tok == '\n' && len > 32U) break;
        if (len > 220U) break;
        if ((tok == '.' || tok == '!' || tok == '?') && len > 48U) break;
    }
    out[len] = '\0';
}

static void chat_showcase(const run_config_t *cfg, const unsigned char *conv_stream, const char *prompt) {
    const model_kind_t models[] = { MODEL_TRIGRAM, MODEL_FOURGRAM, MODEL_OCTAGRAM, MODEL_CASCADE };
    const uint32_t mcount = (uint32_t)(sizeof(models) / sizeof(models[0]));

    printf("\n=== Conversation Test: %s ===\n", prompt);
    for (uint32_t i = 0; i < mcount; ++i) {
        runtime_t rt;
        char out[256];

        if (!runtime_init(&rt, models[i], cfg->slots, cfg->ctx_count)) {
            printf("%s: init failed\n", model_name(models[i]));
            continue;
        }

        for (uint32_t p = 0; p < cfg->bootstrap_passes; ++p) {
            prime_from_text(&rt, conv_stream);
        }

        generate_reply(&rt, prompt, out, sizeof(out));
        printf("%s> %s\n", model_name(models[i]), out);
        runtime_free(&rt);
    }
}

int main(void) {
    const unsigned char *gen = generated_corpus();
    const unsigned char *conv = conversation_corpus();
    size_t ext_len = 0;
    unsigned char *external_conv = load_optional_file("conversation_train.txt", &ext_len);

    const unsigned char *train_stream = external_conv ? external_conv : conv;
    uint32_t train_len = (uint32_t)strlen((const char *)train_stream);
    uint32_t gen_len = (uint32_t)strlen((const char *)gen);

    const run_config_t cfgs[MAX_CONFIGS] = {
        { "tiny", MODEL_CASCADE, 8U, 4096U, 2U, 20000U },
        { "small", MODEL_CASCADE, 12U, 16384U, 4U, 40000U },
        { "mid", MODEL_CASCADE, 16U, 65536U, 6U, 60000U },
        { "lean-speed", MODEL_CASCADE, 10U, 32768U, 3U, 50000U }
    };

    const model_kind_t models[MAX_MODELS] = {
        MODEL_BIGRAM, MODEL_TRIGRAM, MODEL_FOURGRAM, MODEL_OCTAGRAM, MODEL_CASCADE
    };

    bench_result_t results[MAX_CONFIGS * MAX_MODELS];
    uint32_t k = 0;

    printf("next_nothing_lab: binary-only architecture sweep\n");
    printf("training source: %s\n", external_conv ? "conversation_train.txt" : "embedded conversation corpus");
    printf("scoring view: mean(in-domain conversation, out-of-domain generated)\n");

    for (uint32_t c = 0; c < MAX_CONFIGS; ++c) {
        for (uint32_t m = 0; m < MAX_MODELS; ++m) {
            bench_result_t in_domain = run_benchmark(&cfgs[c], models[m], train_stream, train_len, train_stream, train_len);
            bench_result_t out_domain = run_benchmark(&cfgs[c], models[m], train_stream, train_len, gen, gen_len);

            results[k] = in_domain;
            results[k].accuracy = 0.5 * (in_domain.accuracy + out_domain.accuracy);
            results[k].tokens_per_sec = 0.5 * (in_domain.tokens_per_sec + out_domain.tokens_per_sec);
            results[k].memory_mb = in_domain.memory_mb;
            k++;
        }
    }

    score_results(results, k);
    print_results(results, k);

    /* Side-by-side conversation outputs across model variants. */
    chat_showcase(&cfgs[2], train_stream, "How should we scale safely without exploding cost?");
    chat_showcase(&cfgs[2], train_stream, "What is the limiting factor for this architecture?");

    if (external_conv) free(external_conv);
    return 0;
}
