#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

/*
    Binary-first always-on architecture with:
    1) multi-timescale world model
    2) memory hierarchy
    3) enhanced language path
    4) cross-modal binding
    5) plasticity control
    6) active learning policy
    7) background self-distillation
*/

#define EVENT_BITS 1024
#define EVENT_WORDS (EVENT_BITS / 64)
#define MAX_ACTIVE_BITS 128

#define MAX_BLOCKS 64
#define TOP_K_BLOCKS 8

#define MODALITY_COUNT 4
#define CONTEXT_SHORT 8
#define CONTEXT_LONG 32
#define WORKING_RING 128

#define EPISODIC_CAP 384
#define SEMANTIC_CAP 96

#define THOUGHT_RING 2048
#define SPEECH_MAX 256

#define MID_PERIOD 16
#define SLOW_PERIOD 256
#define TUNE_PERIOD 1024
#define REPORT_PERIOD 4096

#define DEMO_MAX_STEPS 140000
#define NGRAM_WARMUP_STEPS 12000

#define ACTION_COUNT 4
#define ACT_EXTERNAL 0
#define ACT_SELF_ROLLOUT 1
#define ACT_EPISODIC_QUERY 2
#define ACT_EXPLORE 3

#define VOCAB " etaoinshrdlucmfwypvbgkqjxz0123456789.,!?-_'\n"
#define VOCAB_SIZE ((int)(sizeof(VOCAB) - 1))
#define MAX_COUNT_10BIT 1023U
#define EVAL_WINDOW 2048

#define BIAS_MIN (-8)
#define BIAS_MAX (7)

typedef enum {
    MOD_VISION = 0,
    MOD_AUDIO = 1,
    MOD_TOUCH = 2,
    MOD_SYMBOLIC = 3
} modality_t;

typedef struct {
    modality_t modality;
    uint64_t bits[EVENT_WORDS];
    uint16_t active_idx[MAX_ACTIVE_BITS];
    uint8_t active_count;
    uint8_t confidence_4bit;
    uint8_t novelty_4bit;
    uint64_t t_bucket;
    int token_id;
} sparse_event_t;

typedef struct {
    uint64_t signature[EVENT_WORDS];
    uint8_t next_token;
    uint8_t strength_4bit;
    uint16_t age;
} episodic_entry_t;

typedef struct {
    uint64_t prototype[EVENT_WORDS];
    int8_t token_hint_4bit[VOCAB_SIZE];
    uint16_t usage;
} semantic_entry_t;

typedef struct {
    int8_t feature_bias_4bit[EVENT_BITS];
    int8_t token_bias_4bit[VOCAB_SIZE];
    uint64_t prototype[EVENT_WORDS];
    int32_t utility;
    uint32_t wins;
    uint32_t misses;
    uint32_t compute_cost;
    uint8_t plasticity_4bit;
    modality_t assignment;
    bool frozen;
} compute_block_t;

typedef struct {
    compute_block_t blocks[MAX_BLOCKS];

    uint8_t working_tokens[WORKING_RING];
    uint8_t short_history[CONTEXT_SHORT];
    uint8_t long_history[CONTEXT_LONG];
    uint16_t working_head;
    uint8_t short_head;
    uint8_t long_head;

    episodic_entry_t episodic[EPISODIC_CAP];
    semantic_entry_t semantic[SEMANTIC_CAP];
    uint16_t episodic_head;
    uint16_t semantic_head;

    uint64_t cross_modal_proto[MODALITY_COUNT][EVENT_WORDS];
    uint32_t modality_demand[MODALITY_COUNT];
    uint32_t modality_error[MODALITY_COUNT];

    uint64_t thought_stream[THOUGHT_RING];
    uint16_t thought_head;

    char speech_buffer[SPEECH_MAX];
    uint16_t speech_len;
    uint64_t last_speech_step;
    bool speaking;

    uint8_t last_prediction;
    uint8_t last_observed;

    uint16_t unigram[VOCAB_SIZE];
    uint16_t bigram[VOCAB_SIZE][VOCAB_SIZE];
    uint16_t trigram[VOCAB_SIZE][VOCAB_SIZE][VOCAB_SIZE];

    uint8_t mix_sparse_4bit;
    uint8_t mix_trigram_4bit;
    uint8_t mix_bigram_4bit;
    uint8_t mix_copy_4bit;
    uint8_t mix_semantic_4bit;

    uint8_t speak_conf_threshold_4bit;
    uint8_t speak_novelty_threshold_4bit;
    uint8_t action_external_bias_4bit;

    uint8_t correct_ring[EVAL_WINDOW];
    uint16_t correct_head;
    uint32_t rolling_correct;
    uint32_t rolling_seen;
    uint64_t total_correct;
    uint64_t total_seen;

    const char *corpus;
    uint32_t corpus_idx;

    uint64_t step;
    bool running;
} runtime_t;

static inline int8_t sat4_add(int8_t x, int8_t delta) {
    int16_t v = (int16_t)x + (int16_t)delta;
    if (v < BIAS_MIN) return BIAS_MIN;
    if (v > BIAS_MAX) return BIAS_MAX;
    return (int8_t)v;
}

static inline uint16_t sat_count_inc(uint16_t v, uint16_t add) {
    uint32_t n = (uint32_t)v + (uint32_t)add;
    if (n > MAX_COUNT_10BIT) return MAX_COUNT_10BIT;
    return (uint16_t)n;
}

static inline uint32_t popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return (uint32_t)__popcnt64(x);
#else
    return (uint32_t)__builtin_popcountll(x);
#endif
}

static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline int token_to_id(char c) {
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        if (VOCAB[i] == c) return i;
    }
    return 0;
}

static inline char id_to_token(int id) {
    if (id < 0 || id >= VOCAB_SIZE) return ' ';
    return VOCAB[id];
}

static inline uint8_t token_class(uint8_t tok) {
    char c = id_to_token((int)tok);
    if (c >= '0' && c <= '9') return 4;
    if (c == ' ' || c == '\n' || c == '\t') return 3;
    if (c == '.' || c == ',' || c == '!' || c == '?' || c == '-' || c == '_' || c == '\'') return 2;
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return 1;
    return 0;
}

static inline void clear_event(sparse_event_t *e) {
    memset(e, 0, sizeof(*e));
}

static inline void set_event_bit(sparse_event_t *e, uint32_t bit) {
    uint32_t w = bit >> 6;
    uint64_t mask = 1ULL << (bit & 63U);
    if ((e->bits[w] & mask) == 0ULL) {
        e->bits[w] |= mask;
        if (e->active_count < MAX_ACTIVE_BITS) {
            e->active_idx[e->active_count++] = (uint16_t)bit;
        }
    }
}

static uint32_t similarity_score(const uint64_t *a, const uint64_t *b) {
    uint32_t s = 0;
    for (int i = 0; i < EVENT_WORDS; ++i) {
        s += popcount64(a[i] & b[i]);
    }
    return s;
}

static void push_token(runtime_t *rt, uint8_t tok) {
    rt->working_tokens[rt->working_head] = tok;
    rt->working_head = (uint16_t)((rt->working_head + 1U) % WORKING_RING);

    rt->short_history[rt->short_head] = tok;
    rt->short_head = (uint8_t)((rt->short_head + 1U) % CONTEXT_SHORT);

    rt->long_history[rt->long_head] = tok;
    rt->long_head = (uint8_t)((rt->long_head + 1U) % CONTEXT_LONG);
}

static uint8_t recent_token(const runtime_t *rt, uint16_t back) {
    uint16_t idx = (uint16_t)((rt->working_head + WORKING_RING - 1U - back) % WORKING_RING);
    return rt->working_tokens[idx];
}

static uint32_t rolling_accuracy_permille(const runtime_t *rt) {
    if (rt->rolling_seen == 0U) return 0U;
    return (rt->rolling_correct * 1000U) / rt->rolling_seen;
}

static void update_correctness_metrics(runtime_t *rt, bool correct) {
    uint8_t prev = rt->correct_ring[rt->correct_head];
    if (rt->rolling_seen >= EVAL_WINDOW) {
        rt->rolling_correct -= prev;
    } else {
        rt->rolling_seen++;
    }

    uint8_t now = (uint8_t)(correct ? 1U : 0U);
    rt->correct_ring[rt->correct_head] = now;
    rt->rolling_correct += now;
    rt->correct_head = (uint16_t)((rt->correct_head + 1U) % EVAL_WINDOW);

    rt->total_seen++;
    rt->total_correct += now;
}

static void update_ngram_counts(runtime_t *rt, uint8_t observed) {
    uint8_t t0 = recent_token(rt, 0);
    uint8_t t1 = recent_token(rt, 1);

    rt->unigram[observed] = sat_count_inc(rt->unigram[observed], 1);
    rt->bigram[t0][observed] = sat_count_inc(rt->bigram[t0][observed], 1);
    rt->trigram[t1][t0][observed] = sat_count_inc(rt->trigram[t1][t0][observed], 1);
}

static void encode_symbolic_event(const runtime_t *rt, sparse_event_t *e, bool long_scale, uint64_t t_bucket) {
    clear_event(e);
    e->modality = MOD_SYMBOLIC;
    e->t_bucket = t_bucket;

    uint32_t window = long_scale ? CONTEXT_LONG : CONTEXT_SHORT;
    for (uint32_t i = 0; i < window; ++i) {
        uint8_t tok = long_scale ? rt->long_history[i % CONTEXT_LONG] : rt->short_history[i % CONTEXT_SHORT];
        uint8_t cls = token_class(tok);
        for (uint32_t j = 0; j < 6U; ++j) {
            uint32_t h = mix32((uint32_t)tok * 131U + i * 37U + j * 97U + (long_scale ? 0x1234567U : 0x89abcdeU));
            set_event_bit(e, h % EVENT_BITS);
            uint32_t hc = mix32((uint32_t)cls * 211U + i * 29U + j * 43U + 0x55aa55aaU);
            set_event_bit(e, hc % EVENT_BITS);
        }
    }

    uint8_t t0 = recent_token(rt, 0);
    uint8_t t1 = recent_token(rt, 1);
    uint8_t t2 = recent_token(rt, 2);
    uint32_t bi = mix32((uint32_t)t0 * 257U + (uint32_t)t1 * 263U + (long_scale ? 71U : 73U));
    uint32_t tri = mix32((uint32_t)t0 * 269U + (uint32_t)t1 * 271U + (uint32_t)t2 * 277U + (long_scale ? 79U : 83U));
    set_event_bit(e, bi % EVENT_BITS);
    set_event_bit(e, tri % EVENT_BITS);

    e->confidence_4bit = 8;
    e->novelty_4bit = 8;
}

static void encode_non_symbolic_stub(const runtime_t *rt, sparse_event_t *e, modality_t m, uint64_t t_bucket) {
    clear_event(e);
    e->modality = m;
    e->t_bucket = t_bucket;

    uint8_t base = recent_token(rt, (uint16_t)m);
    for (uint32_t i = 0; i < 10U; ++i) {
        uint32_t h = mix32((uint32_t)base * 101U + (uint32_t)m * 719U + i * 131U + (uint32_t)(t_bucket & 127ULL));
        set_event_bit(e, h % EVENT_BITS);
    }
}

static void predict_missing_modalities(runtime_t *rt, sparse_event_t events[MODALITY_COUNT]) {
    for (int m = 0; m < MODALITY_COUNT; ++m) {
        if (m == MOD_SYMBOLIC) continue;
        if (events[m].active_count > 0) continue;
        for (int w = 0; w < EVENT_WORDS; ++w) {
            events[m].bits[w] = rt->cross_modal_proto[m][w] & (events[MOD_SYMBOLIC].bits[w] | ~0ULL);
        }
        for (int w = 0; w < EVENT_WORDS; ++w) {
            uint64_t x = events[m].bits[w];
            while (x && events[m].active_count < MAX_ACTIVE_BITS) {
                uint64_t lsb = x & (~x + 1ULL);
                uint32_t bit = (uint32_t)popcount64(lsb - 1ULL);
                events[m].active_idx[events[m].active_count++] = (uint16_t)(w * 64 + bit);
                x &= (x - 1ULL);
            }
        }
        events[m].confidence_4bit = 3;
        events[m].novelty_4bit = 11;
    }
}

static void encode_modal_inputs(runtime_t *rt, sparse_event_t events[MODALITY_COUNT], uint64_t t_bucket) {
    encode_symbolic_event(rt, &events[MOD_SYMBOLIC], false, t_bucket);
    encode_non_symbolic_stub(rt, &events[MOD_VISION], MOD_VISION, t_bucket);
    encode_non_symbolic_stub(rt, &events[MOD_AUDIO], MOD_AUDIO, t_bucket);
    encode_non_symbolic_stub(rt, &events[MOD_TOUCH], MOD_TOUCH, t_bucket);

    if ((rt->step % 19ULL) == 0ULL) {
        clear_event(&events[MOD_VISION]);
        events[MOD_VISION].modality = MOD_VISION;
        events[MOD_VISION].t_bucket = t_bucket;
    }

    predict_missing_modalities(rt, events);
}

static void rank_topk_blocks(const runtime_t *rt, const sparse_event_t *e, int selected[TOP_K_BLOCKS], uint32_t selected_score[TOP_K_BLOCKS]) {
    for (int k = 0; k < TOP_K_BLOCKS; ++k) {
        selected[k] = 0;
        selected_score[k] = 0;
    }

    for (int i = 0; i < MAX_BLOCKS; ++i) {
        const compute_block_t *b = &rt->blocks[i];
        if (b->frozen) continue;

        uint32_t s = similarity_score(e->bits, b->prototype);
        if (b->assignment == e->modality) s += 6U;
        s += (uint32_t)b->plasticity_4bit;

        for (int k = 0; k < TOP_K_BLOCKS; ++k) {
            if (s > selected_score[k]) {
                for (int j = TOP_K_BLOCKS - 1; j > k; --j) {
                    selected_score[j] = selected_score[j - 1];
                    selected[j] = selected[j - 1];
                }
                selected_score[k] = s;
                selected[k] = i;
                break;
            }
        }
    }
}

static int next_external_token(runtime_t *rt) {
    char c = rt->corpus[rt->corpus_idx];
    if (c == '\0') {
        rt->corpus_idx = 0;
        c = rt->corpus[0];
    }
    rt->corpus_idx++;
    return token_to_id(c);
}

static uint32_t event_similarity_to_signature(const sparse_event_t *e, const uint64_t *sig) {
    return similarity_score(e->bits, sig);
}

static void add_copy_bonus(const runtime_t *rt, int32_t score[VOCAB_SIZE]) {
    for (uint16_t i = 0; i < 24U; ++i) {
        uint8_t tok = recent_token(rt, i);
        int32_t boost = 20 - (int32_t)i;
        if (boost < 1) boost = 1;
        score[tok] += boost * (int32_t)rt->mix_copy_4bit;
    }
}

static void add_episodic_bonus(const runtime_t *rt, const sparse_event_t *e, int32_t score[VOCAB_SIZE]) {
    for (uint16_t i = 0; i < EPISODIC_CAP; ++i) {
        const episodic_entry_t *ep = &rt->episodic[i];
        if (ep->strength_4bit == 0) continue;
        uint32_t sim = event_similarity_to_signature(e, ep->signature);
        if (sim > 8U) {
            score[ep->next_token] += ((int32_t)(sim >> 1) + (int32_t)ep->strength_4bit) * (int32_t)rt->mix_semantic_4bit;
        }
    }
}

static void add_semantic_bonus(const runtime_t *rt, const sparse_event_t *e, int32_t score[VOCAB_SIZE]) {
    for (uint16_t i = 0; i < SEMANTIC_CAP; ++i) {
        const semantic_entry_t *se = &rt->semantic[i];
        if (se->usage == 0) continue;
        uint32_t sim = similarity_score(e->bits, se->prototype);
        if (sim > 6U) {
            for (int t = 0; t < VOCAB_SIZE; ++t) {
                score[t] += (int32_t)se->token_hint_4bit[t] * (int32_t)(sim >> 2) * (int32_t)rt->mix_semantic_4bit;
            }
        }
    }
}

static void add_ngram_bonus(const runtime_t *rt, int32_t score[VOCAB_SIZE]) {
    uint8_t t0 = recent_token(rt, 0);
    uint8_t t1 = recent_token(rt, 1);

    for (int tok = 0; tok < VOCAB_SIZE; ++tok) {
        uint16_t tri = rt->trigram[t1][t0][tok];
        uint16_t bi = rt->bigram[t0][tok];
        uint16_t uni = rt->unigram[tok];

        int32_t tri_term = (int32_t)(tri > 63U ? 63U : tri) * (int32_t)rt->mix_trigram_4bit;
        int32_t bi_term = (int32_t)(bi > 63U ? 63U : bi) * (int32_t)rt->mix_bigram_4bit;
        int32_t uni_term = (int32_t)(uni > 63U ? 63U : uni);

        score[tok] += tri_term + (bi_term >> 1) + (uni_term >> 3);
    }
}

static int predict_next_token(
    runtime_t *rt,
    const sparse_event_t *short_e,
    const sparse_event_t *long_e,
    int selected_short[TOP_K_BLOCKS],
    uint32_t selected_short_score[TOP_K_BLOCKS],
    int selected_long[TOP_K_BLOCKS],
    uint32_t selected_long_score[TOP_K_BLOCKS],
    uint8_t *confidence_4bit,
    uint8_t *novelty_4bit
) {
    uint32_t acc = rolling_accuracy_permille(rt);
    int32_t score[VOCAB_SIZE];
    for (int t = 0; t < VOCAB_SIZE; ++t) score[t] = 0;

    for (int k = 0; k < TOP_K_BLOCKS; ++k) {
        compute_block_t *bs = &rt->blocks[selected_short[k]];
        compute_block_t *bl = &rt->blocks[selected_long[k]];
        int32_t gs = 1 + (int32_t)(selected_short_score[k] >> 4);
        int32_t gl = 1 + (int32_t)(selected_long_score[k] >> 4);

        for (int t = 0; t < VOCAB_SIZE; ++t) {
            score[t] += (int32_t)bs->token_bias_4bit[t] * gs * (int32_t)rt->mix_sparse_4bit;
            score[t] += (int32_t)bl->token_bias_4bit[t] * gl * (int32_t)rt->mix_sparse_4bit;
        }
    }

    add_ngram_bonus(rt, score);
    if (acc >= 400U) {
        add_copy_bonus(rt, score);
        add_episodic_bonus(rt, short_e, score);
        add_semantic_bonus(rt, long_e, score);
    }

    int best = 0;
    int second = 0;
    for (int t = 1; t < VOCAB_SIZE; ++t) {
        if (score[t] > score[best]) {
            second = best;
            best = t;
        } else if (t != best && score[t] > score[second]) {
            second = t;
        }
    }

    int32_t margin = score[best] - score[second];
    if (margin < 0) margin = 0;
    if (margin > 15) margin = 15;
    *confidence_4bit = (uint8_t)margin;

    uint32_t seen = 0;
    for (uint16_t i = 0; i < 32U; ++i) {
        if (recent_token(rt, i) == (uint8_t)best) {
            seen++;
        }
    }
    *novelty_4bit = (uint8_t)(seen > 15U ? 0U : (15U - seen));

    if (*confidence_4bit < 5U) {
        uint32_t long_best = selected_long_score[0];
        uint32_t short_best = selected_short_score[0];
        if (long_best > short_best) {
            int fallback = recent_token(rt, 1);
            return fallback;
        }
    }

    return best;
}

static void store_episode(runtime_t *rt, const sparse_event_t *e, uint8_t next_token) {
    episodic_entry_t *ep = &rt->episodic[rt->episodic_head];
    for (int w = 0; w < EVENT_WORDS; ++w) {
        ep->signature[w] = e->bits[w];
    }
    ep->next_token = next_token;
    ep->strength_4bit = 8;
    ep->age = 0;
    rt->episodic_head = (uint16_t)((rt->episodic_head + 1U) % EPISODIC_CAP);
}

static void update_cross_modal_bindings(runtime_t *rt, sparse_event_t events[MODALITY_COUNT]) {
    for (int m = 0; m < MODALITY_COUNT; ++m) {
        if (m == MOD_SYMBOLIC) continue;
        for (int w = 0; w < EVENT_WORDS; ++w) {
            uint64_t co = events[m].bits[w] & events[MOD_SYMBOLIC].bits[w];
            rt->cross_modal_proto[m][w] |= co;
            if ((rt->step % SLOW_PERIOD) == 0ULL) {
                rt->cross_modal_proto[m][w] &= (rt->cross_modal_proto[m][w] | (rt->cross_modal_proto[m][w] >> 1));
            }
        }
    }
}

static uint8_t choose_action(runtime_t *rt, uint8_t confidence_4bit, uint8_t novelty_4bit) {
    (void)confidence_4bit;
    (void)novelty_4bit;
    if (rt->step < NGRAM_WARMUP_STEPS) {
        return ACT_EXTERNAL;
    }

    uint32_t acc = rolling_accuracy_permille(rt);

    if (acc < 850U) {
        return ACT_EXTERNAL;
    }

    int32_t info_gain[ACTION_COUNT] = {0};

    info_gain[ACT_EXTERNAL] = (int32_t)(15U - confidence_4bit) + (int32_t)novelty_4bit + (int32_t)rt->action_external_bias_4bit;
    info_gain[ACT_SELF_ROLLOUT] = (int32_t)confidence_4bit + (acc > 700U ? 5 : -2);
    info_gain[ACT_EPISODIC_QUERY] = (int32_t)(novelty_4bit / 2U) + 4;
    info_gain[ACT_EXPLORE] = (int32_t)(15U - confidence_4bit) + 1;

    int best = 0;
    for (int i = 1; i < ACTION_COUNT; ++i) {
        if (info_gain[i] > info_gain[best]) best = i;
    }

    if ((rt->step % 97ULL) == 0ULL && acc > 600U) {
        best = ACT_EXPLORE;
    }

    return (uint8_t)best;
}

static uint8_t episodic_query_token(const runtime_t *rt, const sparse_event_t *e) {
    uint32_t best_sim = 0;
    uint8_t tok = (uint8_t)token_to_id(' ');
    for (uint16_t i = 0; i < EPISODIC_CAP; ++i) {
        const episodic_entry_t *ep = &rt->episodic[i];
        if (ep->strength_4bit == 0) continue;
        uint32_t sim = event_similarity_to_signature(e, ep->signature);
        if (sim > best_sim) {
            best_sim = sim;
            tok = ep->next_token;
        }
    }
    return tok;
}

static int8_t learning_delta(bool correct, uint8_t surprise_4bit, int32_t utility) {
    int8_t base = correct ? 1 : -1;
    int8_t gate = 1;
    if (!correct && surprise_4bit > 7U) gate = 3;
    else if (surprise_4bit > 4U) gate = 2;
    if (utility < 0) gate++;
    if (gate > 4) gate = 4;
    return (int8_t)(base * gate);
}

static void update_block_from_event(compute_block_t *b, const sparse_event_t *e, int8_t delta) {
    for (uint8_t i = 0; i < e->active_count; ++i) {
        uint32_t idx = e->active_idx[i];
        b->feature_bias_4bit[idx] = sat4_add(b->feature_bias_4bit[idx], delta);
        uint32_t w = idx >> 6;
        uint64_t mask = 1ULL << (idx & 63U);
        if (b->feature_bias_4bit[idx] > 0) {
            b->prototype[w] |= mask;
        } else {
            b->prototype[w] &= ~mask;
        }
    }
}

static void learn_step(
    runtime_t *rt,
    const sparse_event_t *short_e,
    int selected_short[TOP_K_BLOCKS],
    int selected_long[TOP_K_BLOCKS],
    uint8_t predicted_token,
    uint8_t observed_token,
    uint8_t confidence_4bit,
    uint8_t novelty_4bit
) {
    bool correct = (predicted_token == observed_token);
    uint8_t surprise = (uint8_t)(correct ? 1U : (8U + novelty_4bit / 2U));

    for (int k = 0; k < TOP_K_BLOCKS; ++k) {
        compute_block_t *bs = &rt->blocks[selected_short[k]];
        compute_block_t *bl = &rt->blocks[selected_long[k]];

        int8_t ds = learning_delta(correct, surprise, bs->utility);
        int8_t dl = learning_delta(correct, surprise, bl->utility);

        bs->token_bias_4bit[observed_token] = sat4_add(bs->token_bias_4bit[observed_token], ds);
        bl->token_bias_4bit[observed_token] = sat4_add(bl->token_bias_4bit[observed_token], dl);
        if (!correct) {
            bs->token_bias_4bit[predicted_token] = sat4_add(bs->token_bias_4bit[predicted_token], (int8_t)(-ds));
            bl->token_bias_4bit[predicted_token] = sat4_add(bl->token_bias_4bit[predicted_token], (int8_t)(-dl));
        }

        update_block_from_event(bs, short_e, ds);
        update_block_from_event(bl, short_e, dl);

        bs->wins += (uint32_t)correct;
        bs->misses += (uint32_t)(!correct);
        bs->compute_cost++;
        bs->utility = (int32_t)bs->wins - (int32_t)bs->misses - (int32_t)(bs->compute_cost >> 2);
        bs->plasticity_4bit = (uint8_t)(confidence_4bit > 12U ? 2U : (8U + novelty_4bit / 2U));

        bl->wins += (uint32_t)correct;
        bl->misses += (uint32_t)(!correct);
        bl->compute_cost++;
        bl->utility = (int32_t)bl->wins - (int32_t)bl->misses - (int32_t)(bl->compute_cost >> 2);
        bl->plasticity_4bit = (uint8_t)(confidence_4bit > 12U ? 2U : (8U + novelty_4bit / 2U));
    }

    store_episode(rt, short_e, observed_token);
    update_ngram_counts(rt, observed_token);
    update_correctness_metrics(rt, correct);
    rt->modality_demand[MOD_SYMBOLIC]++;
    rt->modality_error[MOD_SYMBOLIC] += (uint32_t)(!correct);
}

static void update_thought_stream(runtime_t *rt, uint8_t predicted_token, uint8_t confidence_4bit, uint8_t novelty_4bit, uint8_t action) {
    uint64_t packet = 0ULL;
    packet |= (rt->step & 0x00000000ffffffffULL) << 24;
    packet |= ((uint64_t)predicted_token & 0xFFULL) << 16;
    packet |= ((uint64_t)confidence_4bit & 0x0FULL) << 12;
    packet |= ((uint64_t)novelty_4bit & 0x0FULL) << 8;
    packet |= ((uint64_t)action & 0x0FULL) << 4;
    packet |= ((uint64_t)rt->last_observed & 0x0FULL);
    rt->thought_stream[rt->thought_head] = packet;
    rt->thought_head = (uint16_t)((rt->thought_head + 1U) % THOUGHT_RING);
}

static void maybe_speak(runtime_t *rt, uint8_t predicted_token, uint8_t confidence_4bit, uint8_t novelty_4bit, uint8_t action) {
    uint32_t acc = rolling_accuracy_permille(rt);

    bool open_gate =
        !rt->speaking &&
        confidence_4bit >= rt->speak_conf_threshold_4bit &&
        novelty_4bit <= rt->speak_novelty_threshold_4bit &&
        action != ACT_EXPLORE &&
        acc >= 850U &&
        (rt->step - rt->last_speech_step) > 256ULL;

    if (open_gate) {
        rt->speaking = true;
        rt->speech_len = 0;
        memset(rt->speech_buffer, 0, sizeof(rt->speech_buffer));
    }

    if (!rt->speaking) return;

    if (rt->speech_len < (SPEECH_MAX - 1U)) {
        rt->speech_buffer[rt->speech_len++] = id_to_token(predicted_token);
        rt->speech_buffer[rt->speech_len] = '\0';
    }

    char c = id_to_token(predicted_token);
    bool close_gate =
        rt->speech_len >= 42U ||
        ((c == '.' || c == '!' || c == '?' || c == '\n') && rt->speech_len > 8U);

    if (close_gate) {
        printf("say:%s\n", rt->speech_buffer);
        rt->last_speech_step = rt->step;
        rt->speaking = false;
        rt->speech_len = 0;
    }
}

static void auto_tune_parameters(runtime_t *rt) {
    uint32_t acc = rolling_accuracy_permille(rt);

    if (acc < 500U) {
        if (rt->mix_trigram_4bit < 15U) rt->mix_trigram_4bit++;
        if (rt->mix_bigram_4bit < 15U) rt->mix_bigram_4bit++;
        rt->mix_sparse_4bit = 0U;
        rt->mix_semantic_4bit = 0U;
        if (rt->mix_copy_4bit > 1U) rt->mix_copy_4bit--;
        if (rt->action_external_bias_4bit < 15U) rt->action_external_bias_4bit++;
        rt->speak_conf_threshold_4bit = 14U;
        rt->speak_novelty_threshold_4bit = 6U;
    } else if (acc < 700U) {
        if (rt->mix_trigram_4bit < 15U) rt->mix_trigram_4bit++;
        if (rt->mix_copy_4bit > 2U) rt->mix_copy_4bit--;
        if (rt->action_external_bias_4bit < 15U) rt->action_external_bias_4bit++;
        rt->speak_conf_threshold_4bit = 13U;
        rt->speak_novelty_threshold_4bit = 7U;
    } else {
        if (rt->mix_sparse_4bit < 15U) rt->mix_sparse_4bit++;
        if (rt->action_external_bias_4bit > 4U) rt->action_external_bias_4bit--;
        if (rt->speak_conf_threshold_4bit > 10U) rt->speak_conf_threshold_4bit--;
        if (rt->speak_novelty_threshold_4bit < 9U) rt->speak_novelty_threshold_4bit++;
    }
}

static void maybe_report(runtime_t *rt) {
    if ((rt->step % REPORT_PERIOD) != 0ULL) return;

    uint32_t rolling = rolling_accuracy_permille(rt);
    uint32_t global = 0;
    if (rt->total_seen > 0ULL) {
        global = (uint32_t)((rt->total_correct * 1000ULL) / rt->total_seen);
    }

    printf(
        "stat step=%llu roll=%.1f%% global=%.1f%% mix[s=%u t=%u b=%u c=%u m=%u] ext=%u\n",
        (unsigned long long)rt->step,
        (double)rolling / 10.0,
        (double)global / 10.0,
        rt->mix_sparse_4bit,
        rt->mix_trigram_4bit,
        rt->mix_bigram_4bit,
        rt->mix_copy_4bit,
        rt->mix_semantic_4bit,
        rt->action_external_bias_4bit
    );
}

static void reassign_blocks(runtime_t *rt) {
    int target_modality = MOD_SYMBOLIC;
    uint32_t best_pressure = 0;

    for (int m = 0; m < MODALITY_COUNT; ++m) {
        uint32_t p = rt->modality_demand[m] + (rt->modality_error[m] << 3);
        if (p > best_pressure) {
            best_pressure = p;
            target_modality = m;
        }
        rt->modality_demand[m] = 0;
        rt->modality_error[m] = 0;
    }

    int weak = 0;
    int strong = 0;
    for (int i = 1; i < MAX_BLOCKS; ++i) {
        if (rt->blocks[i].utility < rt->blocks[weak].utility) weak = i;
        if (rt->blocks[i].utility > rt->blocks[strong].utility) strong = i;
    }

    rt->blocks[strong].frozen = true;
    rt->blocks[weak].assignment = (modality_t)target_modality;
    rt->blocks[weak].wins = 0;
    rt->blocks[weak].misses = 0;
    rt->blocks[weak].compute_cost = 0;
    rt->blocks[weak].frozen = false;
}

static void distill_semantic(runtime_t *rt) {
    semantic_entry_t *se = &rt->semantic[rt->semantic_head];
    memset(se, 0, sizeof(*se));

    uint32_t total = 0;
    for (uint16_t i = 0; i < EPISODIC_CAP; ++i) {
        episodic_entry_t *ep = &rt->episodic[i];
        if (ep->strength_4bit == 0) continue;
        for (int w = 0; w < EVENT_WORDS; ++w) {
            se->prototype[w] |= ep->signature[w];
        }
        se->token_hint_4bit[ep->next_token] = sat4_add(se->token_hint_4bit[ep->next_token], 1);
        ep->age++;
        if (ep->age > 512U && ep->strength_4bit > 0) {
            ep->strength_4bit--;
            ep->age = 0;
        }
        total++;
        if (total > 24U) break;
    }

    se->usage = (uint16_t)total;
    rt->semantic_head = (uint16_t)((rt->semantic_head + 1U) % SEMANTIC_CAP);
}

static void mid_timescale_tick(runtime_t *rt, sparse_event_t events[MODALITY_COUNT]) {
    update_cross_modal_bindings(rt, events);
    reassign_blocks(rt);
}

static void slow_timescale_tick(runtime_t *rt) {
    distill_semantic(rt);
}

static uint64_t monotonic_tick_ms(void) {
    return (uint64_t)(clock() * 1000 / CLOCKS_PER_SEC);
}

static void bootstrap_language(runtime_t *rt, uint32_t passes) {
    for (uint32_t p = 0; p < passes; ++p) {
        for (uint32_t i = 0; rt->corpus[i] != '\0'; ++i) {
            uint8_t tok = (uint8_t)token_to_id(rt->corpus[i]);
            update_ngram_counts(rt, tok);
            push_token(rt, tok);
        }
    }
}

static void init_runtime(runtime_t *rt) {
    memset(rt, 0, sizeof(*rt));
    rt->running = true;
    rt->corpus =
        "the quick brown fox jumps over the lazy dog. "
        "real time agents should keep predicting and adapting. "
        "cheap local rules can still build useful world models. "
        "binary first cognition can remain active without speaking. ";

    uint8_t space = (uint8_t)token_to_id(' ');
    for (int i = 0; i < WORKING_RING; ++i) rt->working_tokens[i] = space;
    for (int i = 0; i < CONTEXT_SHORT; ++i) rt->short_history[i] = space;
    for (int i = 0; i < CONTEXT_LONG; ++i) rt->long_history[i] = space;
    rt->last_prediction = space;
    rt->last_observed = space;

    rt->mix_sparse_4bit = 6;
    rt->mix_trigram_4bit = 12;
    rt->mix_bigram_4bit = 10;
    rt->mix_copy_4bit = 3;
    rt->mix_semantic_4bit = 1;

    rt->speak_conf_threshold_4bit = 13;
    rt->speak_novelty_threshold_4bit = 7;
    rt->action_external_bias_4bit = 12;

    for (int i = 0; i < MAX_BLOCKS; ++i) {
        rt->blocks[i].assignment = (i < (MAX_BLOCKS * 3 / 4)) ? MOD_SYMBOLIC : (modality_t)(i % MODALITY_COUNT);
        rt->blocks[i].plasticity_4bit = 8;
    }

    bootstrap_language(rt, 2U);
}

int main(void) {
    static runtime_t rt;
    init_runtime(&rt);

    while (rt.running) {
        sparse_event_t events[MODALITY_COUNT];
        sparse_event_t long_event;

        uint64_t t = monotonic_tick_ms();
        encode_modal_inputs(&rt, events, t >> 4);
        encode_symbolic_event(&rt, &long_event, true, t >> 3);

        int selected_short[TOP_K_BLOCKS];
        uint32_t selected_short_score[TOP_K_BLOCKS];
        int selected_long[TOP_K_BLOCKS];
        uint32_t selected_long_score[TOP_K_BLOCKS];

        rank_topk_blocks(&rt, &events[MOD_SYMBOLIC], selected_short, selected_short_score);
        rank_topk_blocks(&rt, &long_event, selected_long, selected_long_score);

        uint8_t conf = 0;
        uint8_t nov = 0;
        uint8_t pred = (uint8_t)predict_next_token(
            &rt,
            &events[MOD_SYMBOLIC],
            &long_event,
            selected_short,
            selected_short_score,
            selected_long,
            selected_long_score,
            &conf,
            &nov
        );

        uint8_t action = choose_action(&rt, conf, nov);
        uint8_t observed;
        if (action == ACT_SELF_ROLLOUT) {
            observed = pred;
        } else if (action == ACT_EPISODIC_QUERY) {
            observed = episodic_query_token(&rt, &events[MOD_SYMBOLIC]);
        } else if (action == ACT_EXPLORE) {
            observed = (uint8_t)(mix32((uint32_t)rt.step + 0x10203040U) % (uint32_t)VOCAB_SIZE);
        } else {
            observed = (uint8_t)next_external_token(&rt);
        }

        events[MOD_SYMBOLIC].token_id = observed;
        events[MOD_SYMBOLIC].confidence_4bit = conf;
        events[MOD_SYMBOLIC].novelty_4bit = nov;

        learn_step(&rt, &events[MOD_SYMBOLIC], selected_short, selected_long, pred, observed, conf, nov);

        rt.last_prediction = pred;
        rt.last_observed = observed;
        push_token(&rt, observed);
        update_thought_stream(&rt, pred, conf, nov, action);
        maybe_speak(&rt, pred, conf, nov, action);

        if ((rt.step % MID_PERIOD) == 0ULL) {
            mid_timescale_tick(&rt, events);
        }
        if ((rt.step % SLOW_PERIOD) == 0ULL) {
            slow_timescale_tick(&rt);
        }
        if ((rt.step % TUNE_PERIOD) == 0ULL) {
            auto_tune_parameters(&rt);
        }
        maybe_report(&rt);

        rt.step++;
        if (DEMO_MAX_STEPS > 0 && rt.step > DEMO_MAX_STEPS) {
            rt.running = false;
        }
    }

    printf("runtime finished at step=%llu\n", (unsigned long long)rt.step);
    return 0;
}
