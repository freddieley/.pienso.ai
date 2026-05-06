#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
    Retry v2: clean always-on, byte-level online model.
    Predictor: PPM-style cascade — octagram (8-gram) -> fourgram (4-gram)
    -> trigram -> bigram. Highest-order table that has seen the context wins.
    Training: always observe the corpus stream (never own predictions).
*/

#define VOCAB_SIZE 256
#define WORKING_RING 512
#define THOUGHT_RING 4096
#define SPEECH_MAX 256

#define MODALITY_COUNT 4
#define BLOCK_COUNT 32

#define TRI_SLOTS 16
#define TRI_CTX_COUNT 65536U

#define REPORT_PERIOD 2048ULL
#define REASSIGN_PERIOD 2048ULL
#define DEMO_MAX_STEPS 0ULL

typedef enum { MOD_VISION=0, MOD_AUDIO=1, MOD_TOUCH=2, MOD_SYMBOLIC=3 } modality_t;

typedef struct { uint8_t token; uint16_t count; } tri_slot_t;
typedef struct { tri_slot_t slot[TRI_SLOTS]; } tri_bucket_t;
typedef struct { modality_t assignment; int32_t utility; uint32_t demand; uint32_t errors; } block_t;

typedef struct {
    uint32_t unigram[VOCAB_SIZE];
    uint16_t bigram[VOCAB_SIZE][VOCAB_SIZE];
    tri_bucket_t trigram[TRI_CTX_COUNT];
    tri_bucket_t fourgram[TRI_CTX_COUNT];
    tri_bucket_t octagram[TRI_CTX_COUNT];

    uint8_t working[WORKING_RING];
    uint16_t working_head;

    uint64_t thought[THOUGHT_RING];
    uint16_t thought_head;

    block_t blocks[BLOCK_COUNT];

    uint8_t speak_conf_threshold;
    uint16_t speak_min_acc_permille;
    bool speaking;
    char speech_buffer[SPEECH_MAX];
    uint16_t speech_len;
    uint64_t last_speech_step;

    uint8_t correct_ring[4096];
    uint16_t correct_head;
    uint32_t rolling_correct;
    uint32_t rolling_seen;
    uint64_t total_correct;
    uint64_t total_seen;

    const unsigned char *stream;
    uint32_t stream_idx;

    uint8_t last_pred;
    uint8_t last_obs;
    uint64_t step;
    bool running;
} runtime_t;

/* ---- helpers ---- */

static inline uint16_t sat_u16_add(uint16_t v, uint16_t add) {
    uint32_t n = (uint32_t)v + (uint32_t)add;
    return (uint16_t)(n > 65535U ? 65535U : n);
}
static inline uint32_t rolling_acc_permille(const runtime_t *rt) {
    return rt->rolling_seen == 0U ? 0U : (rt->rolling_correct * 1000U) / rt->rolling_seen;
}
static inline uint8_t recent_tok(const runtime_t *rt, uint16_t back) {
    return rt->working[(rt->working_head + WORKING_RING - 1U - back) % WORKING_RING];
}
static void push_tok(runtime_t *rt, uint8_t tok) {
    rt->working[rt->working_head] = tok;
    rt->working_head = (uint16_t)((rt->working_head + 1U) % WORKING_RING);
}
static void update_metrics(runtime_t *rt, bool correct) {
    uint8_t prev = rt->correct_ring[rt->correct_head];
    if (rt->rolling_seen >= 4096U) rt->rolling_correct -= prev;
    else rt->rolling_seen++;
    uint8_t now = correct ? 1U : 0U;
    rt->correct_ring[rt->correct_head] = now;
    rt->correct_head = (uint16_t)((rt->correct_head + 1U) % 4096U);
    rt->rolling_correct += now;
    rt->total_seen++;
    rt->total_correct += now;
}

/* ---- context hashing ---- */

static inline uint16_t tri_ctx(uint8_t t2, uint8_t t1) {
    return (uint16_t)(((uint16_t)t2 << 8) | t1);
}
static inline uint16_t quad_ctx(uint8_t t3, uint8_t t2, uint8_t t1) {
    return (uint16_t)(((uint16_t)t3 * 257U) ^ ((uint16_t)t2 << 8) ^ t1);
}
/* FNV-1a over 7 bytes of context -> 16-bit bucket index. */
static inline uint16_t oct_ctx(uint8_t t7, uint8_t t6, uint8_t t5, uint8_t t4,
                                uint8_t t3, uint8_t t2, uint8_t t1) {
    uint32_t h = 2166136261U;
    h ^= t7; h *= 16777619U;
    h ^= t6; h *= 16777619U;
    h ^= t5; h *= 16777619U;
    h ^= t4; h *= 16777619U;
    h ^= t3; h *= 16777619U;
    h ^= t2; h *= 16777619U;
    h ^= t1; h *= 16777619U;
    return (uint16_t)(h ^ (h >> 16));
}

/* ---- bucket ops (same structure for all n-gram orders) ---- */

static void bucket_inc(tri_bucket_t *b, uint8_t next) {
    int free_i = -1, min_i = 0;
    uint16_t min_c = b->slot[0].count;
    for (int i = 0; i < TRI_SLOTS; ++i) {
        if (b->slot[i].count == 0U) { if (free_i < 0) free_i = i; }
        if (b->slot[i].count > 0U && b->slot[i].token == next) {
            b->slot[i].count = sat_u16_add(b->slot[i].count, 1U);
            return;
        }
        if (b->slot[i].count < min_c) { min_c = b->slot[i].count; min_i = i; }
    }
    if (free_i >= 0) { b->slot[free_i].token = next; b->slot[free_i].count = 1U; return; }
    if (b->slot[min_i].count > 1U) b->slot[min_i].count--;
    else { b->slot[min_i].token = next; b->slot[min_i].count = 1U; }
}

/* Returns false if bucket empty; otherwise fills tok_out/conf_out and returns true. */
static bool bucket_predict(const tri_bucket_t *b, uint8_t *tok_out, uint8_t *conf_out) {
    uint32_t total = 0, bc = 0;
    int best = 0;
    for (int i = 0; i < TRI_SLOTS; ++i) {
        total += b->slot[i].count;
        if (b->slot[i].count > bc) { bc = b->slot[i].count; best = b->slot[i].token; }
    }
    if (total == 0) return false;
    *tok_out = (uint8_t)best;
    uint32_t r = (bc * 255U) / total;
    *conf_out = (uint8_t)(r > 255U ? 255U : r);
    return true;
}

/* ---- language model update ---- */

static void update_language(runtime_t *rt, uint8_t observed) {
    uint8_t t1 = recent_tok(rt, 0), t2 = recent_tok(rt, 1), t3 = recent_tok(rt, 2);
    uint8_t t4 = recent_tok(rt, 3), t5 = recent_tok(rt, 4);
    uint8_t t6 = recent_tok(rt, 5), t7 = recent_tok(rt, 6);

    rt->unigram[observed]++;
    rt->bigram[t1][observed] = sat_u16_add(rt->bigram[t1][observed], 1U);
    bucket_inc(&rt->trigram[tri_ctx(t2, t1)], observed);
    bucket_inc(&rt->fourgram[quad_ctx(t3, t2, t1)], observed);
    bucket_inc(&rt->octagram[oct_ctx(t7, t6, t5, t4, t3, t2, t1)], observed);
}

/* ---- PPM-style cascade predictor ---- */

static uint8_t predict_token(const runtime_t *rt, uint8_t *conf_out) {
    uint8_t t1 = recent_tok(rt, 0), t2 = recent_tok(rt, 1), t3 = recent_tok(rt, 2);
    uint8_t t4 = recent_tok(rt, 3), t5 = recent_tok(rt, 4);
    uint8_t t6 = recent_tok(rt, 5), t7 = recent_tok(rt, 6);

    uint8_t tok = 32U, conf = 0;

    if (bucket_predict(&rt->octagram[oct_ctx(t7,t6,t5,t4,t3,t2,t1)], &tok, &conf))
        { *conf_out = conf; return tok; }
    if (bucket_predict(&rt->fourgram[quad_ctx(t3,t2,t1)], &tok, &conf))
        { *conf_out = conf; return tok; }
    if (bucket_predict(&rt->trigram[tri_ctx(t2,t1)], &tok, &conf))
        { *conf_out = conf; return tok; }

    /* Bigram fallback */
    uint32_t btotal = 0; uint16_t bc = 0; int best = 32;
    for (int t = 0; t < VOCAB_SIZE; ++t) {
        uint16_t v = rt->bigram[t1][t];
        if (v > bc) { bc = v; best = t; }
        btotal += v;
    }
    uint32_t r = btotal > 0 ? (bc * 255U) / btotal : 0U;
    *conf_out = (uint8_t)(r > 255U ? 255U : r);
    return (uint8_t)best;
}

/* ---- corpus stream ---- */

static uint8_t next_external(runtime_t *rt) {
    unsigned char c = rt->stream[rt->stream_idx];
    if (c == '\0') { rt->stream_idx = 0; c = rt->stream[0]; }
    rt->stream_idx++;
    return (uint8_t)c;
}

/* ---- thought stream ---- */

static void write_thought(runtime_t *rt, uint8_t pred, uint8_t obs, uint8_t conf) {
    uint64_t p = ((rt->step & 0xFFFFFFFFULL) << 24) |
                 ((uint64_t)pred << 16) | ((uint64_t)obs << 8) | (uint64_t)conf;
    rt->thought[rt->thought_head] = p;
    rt->thought_head = (uint16_t)((rt->thought_head + 1U) % THOUGHT_RING);
}

/* ---- speech gate ---- */

static void maybe_speak(runtime_t *rt, uint8_t pred, uint8_t conf) {
    uint32_t acc = rolling_acc_permille(rt);
    bool open = !rt->speaking &&
                acc >= rt->speak_min_acc_permille &&
                conf >= rt->speak_conf_threshold &&
                (rt->step - rt->last_speech_step) > 256ULL;
    if (open) { rt->speaking = true; rt->speech_len = 0; memset(rt->speech_buffer, 0, sizeof(rt->speech_buffer)); }
    if (!rt->speaking) return;

    if (rt->speech_len < (SPEECH_MAX - 1U)) {
        char c = (char)pred;
        if ((unsigned char)c < 32U && c != '\n' && c != '\t') c = ' ';
        rt->speech_buffer[rt->speech_len++] = c;
        rt->speech_buffer[rt->speech_len] = '\0';
    }
    char last = rt->speech_buffer[rt->speech_len - 1U];
    bool close = rt->speech_len >= 80U ||
        ((last == '.' || last == '!' || last == '?' || last == '\n') && rt->speech_len > 12U);
    if (close) {
        printf("say:%s\n", rt->speech_buffer);
        rt->speaking = false; rt->speech_len = 0; rt->last_speech_step = rt->step;
    }
}

/* ---- resource recycling ---- */

static void reassign_blocks(runtime_t *rt) {
    int weakest = 0, strongest = 0;
    for (int i = 1; i < BLOCK_COUNT; ++i) {
        if (rt->blocks[i].utility < rt->blocks[weakest].utility) weakest = i;
        if (rt->blocks[i].utility > rt->blocks[strongest].utility) strongest = i;
    }
    int target = MOD_SYMBOLIC; uint32_t pressure = 0U;
    for (int m = 0; m < MODALITY_COUNT; ++m) {
        uint32_t p = rt->blocks[m].demand + (rt->blocks[m].errors << 2);
        if (p > pressure) { pressure = p; target = m; }
    }
    rt->blocks[strongest].utility++;
    rt->blocks[weakest].assignment = (modality_t)target;
    rt->blocks[weakest].utility = 0;
    for (int m = 0; m < MODALITY_COUNT; ++m) { rt->blocks[m].demand = 0; rt->blocks[m].errors = 0; }
}

/* ---- bootstrap ---- */

static void bootstrap(runtime_t *rt, uint32_t passes) {
    for (uint32_t p = 0; p < passes; ++p) {
        for (uint32_t i = 0; rt->stream[i] != '\0'; ++i) {
            uint8_t tok = rt->stream[i];
            update_language(rt, tok);
            push_tok(rt, tok);
        }
    }
}

/* ---- init ---- */

static void init_runtime(runtime_t *rt) {
    memset(rt, 0, sizeof(*rt));

    static const unsigned char corpus[] =
        /* nature and science */
        "the sun rises in the east and sets in the west every single day. "
        "water flows downhill following gravity toward the lowest point always. "
        "trees grow slowly over many years reaching toward the light above. "
        "birds migrate south in autumn and return north each spring reliably. "
        "the ocean covers most of the surface of the earth we live on. "
        "mountains are formed by the movement of tectonic plates over time. "
        "clouds carry moisture that falls as rain or snow on the land below. "
        "rivers carve valleys as they flow from mountains to the distant sea. "
        "animals adapt to their environments through generations of slow change. "
        "plants convert sunlight into energy through photosynthesis every day. "
        /* language and mind */
        "the quick brown fox jumps over the lazy dog near the river bank. "
        "real time language should be fluent, adaptive, and always stable. "
        "always on systems keep predicting even when they remain quite silent. "
        "cheap cpu-native rules can still learn coherent transitions very well. "
        "language is the primary tool humans use to share ideas and knowledge. "
        "words carry meaning shaped by context, tone, and shared experience here. "
        "a sentence must have a subject and a predicate to be complete always. "
        "reading expands vocabulary and sharpens the capacity for deep thought. "
        "the brain forms patterns from repeated exposure to consistent input data. "
        "memory links new information to existing structures already stored there. "
        /* everyday life */
        "every morning people wake up and prepare for the long day ahead. "
        "breakfast gives the body energy to start the work of the day well. "
        "people travel by car, bus, train, or bicycle to reach their jobs daily. "
        "offices and schools are places where people gather to work and learn. "
        "in the evening families often share a meal and talk about their day. "
        "cities are busy places full of people moving in all directions always. "
        "markets sell food, clothing, tools, and many other very useful things. "
        "music fills the air in cafes, parks, streets, and homes every day. "
        "technology has changed the way people communicate with each other now. "
        "a good night of sleep helps the brain consolidate what it has learned. "
        /* time and sequence */
        "first you plan, then you act, then you observe what happened next here. "
        "each second contains a thousand small decisions made without much thought. "
        "patterns repeat across time at many different scales and long durations. "
        "the past shapes the present and the present shapes the future always. "
        "cycles of day and night give structure to the whole rhythm of life. "
        "seasons change from warm to cold and back to warm again each year. "
        "history records the sequence of events that brought us to this point. "
        "clocks divide time into hours, minutes, and seconds for coordination. "
        "deadlines create urgency and help people focus on completing their tasks. "
        "routine makes complex behavior automatic and frees the mind for new work. "
        /* computation and systems */
        "a processor executes instructions one after another at very high speed. "
        "memory stores data that the processor can read and write at any time. "
        "algorithms are step-by-step procedures for solving well-defined problems. "
        "feedback loops allow systems to adjust their behavior based on results. "
        "sparse representations encode information using only a few active units. "
        "prediction errors drive learning by signaling when expectations fail here. "
        "efficient systems do more work with less energy and fewer total operations. "
        "binary choices are the foundation of all digital computation used today. "
        "a model that generalizes well performs on data it has never seen before. "
        "online learning updates the model with each new observation as it arrives. "
        /* reasoning and logic */
        "if the premise is true and the argument is valid the conclusion holds well. "
        "evidence should guide belief and belief should be updated by new evidence. "
        "a hypothesis is a testable prediction derived from a broader theory always. "
        "correlation does not imply causation and careful analysis is always required. "
        "clear thinking requires precise language and well-defined concepts every time. "
        "contradictions reveal errors in reasoning and must be resolved very carefully. "
        "an analogy helps explain an unfamiliar idea using a familiar one instead. "
        "categories help organize the world into manageable groups and many classes. "
        "exceptions to a rule reveal the limits of its scope and its application. "
        "the simplest explanation consistent with the evidence is often the best one. "
        /* social and human */
        "trust is built slowly through consistent and honest behavior over long time. "
        "cooperation allows groups to accomplish more than individuals could ever alone. "
        "conflict arises when the goals or values of different people sharply clash. "
        "empathy is the ability to understand and share the feelings of another person. "
        "culture is the shared set of values, practices, and stories of every group. "
        "teaching passes knowledge from one generation to the next in a reliable way. "
        "children learn language by hearing it spoken in full context all around them. "
        "stories create shared meaning and help communities understand themselves well. "
        "art expresses what language alone cannot always fully capture or ever convey. "
        "play is a form of learning that feels effortless and naturally very rewarding. "
        /* motion and action */
        "objects in motion tend to remain in motion unless a force acts upon them. "
        "energy is required to change the state or velocity of any moving object now. "
        "walking is a controlled fall where balance is constantly being recovered again. "
        "the hand moves to where the eye predicts the target will be located next. "
        "smooth motion results from many small corrections applied in careful sequence. "
        "athletes train their bodies to execute complex movements completely automatically. "
        "the fastest path between two points is a straight line on perfectly flat ground. "
        "tools extend the reach of the human hand and amplify its natural force greatly. "
        "machines transform energy from one form into a more useful form always needed. "
        "practice builds the neural pathways that make skilled actions feel effortless. "
        /* goals and problems */
        "every goal requires a plan and every plan requires a clear sequence of steps. "
        "obstacles are the difference between where you are and where you want to be. "
        "breaking a large problem into smaller parts makes it much easier to solve well. "
        "the first solution is rarely the best and iteration always improves every design. "
        "success depends on choosing the right goal as much as the right method always. "
        "failure reveals information that success often hides from the careful observer. "
        "resources are limited and allocation decisions determine exactly what gets done. "
        "priorities clarify which actions matter most when time is very short always here. "
        "measurement tells you whether you are moving toward or away from the main goal. "
        "persistence in the face of difficulty separates real achievement from mere attempt. "
        "\n";

    rt->stream = corpus;
    rt->running = true;
    rt->speak_conf_threshold = 100;
    rt->speak_min_acc_permille = 880;

    for (int i = 0; i < BLOCK_COUNT; ++i) {
        rt->blocks[i].assignment = (modality_t)(i % MODALITY_COUNT);
        rt->blocks[i].utility = 0;
    }
    bootstrap(rt, 8U);
}

/* ---- reporting ---- */

static void report_stats(const runtime_t *rt) {
    uint32_t roll = rolling_acc_permille(rt);
    uint32_t global = rt->total_seen > 0ULL
        ? (uint32_t)((rt->total_correct * 1000ULL) / rt->total_seen) : 0U;
    printf("stat step=%llu roll=%.1f%% global=%.1f%% speak[conf>=%u acc>=%u]\n",
        (unsigned long long)rt->step,
        (double)roll / 10.0,
        (double)global / 10.0,
        rt->speak_conf_threshold,
        rt->speak_min_acc_permille);
}

/* ---- main loop ---- */

int main(void) {
    static runtime_t rt;
    init_runtime(&rt);

    while (rt.running) {
        uint8_t conf = 0;
        uint8_t pred = predict_token(&rt, &conf);
        uint8_t obs  = next_external(&rt);

        bool correct = (pred == obs);
        update_metrics(&rt, correct);
        update_language(&rt, obs);
        push_tok(&rt, obs);

        rt.blocks[MOD_SYMBOLIC].demand++;
        rt.blocks[MOD_SYMBOLIC].errors += (uint32_t)(!correct);
        rt.last_pred = pred;
        rt.last_obs  = obs;

        write_thought(&rt, pred, obs, conf);
        maybe_speak(&rt, pred, conf);

        if ((rt.step % REASSIGN_PERIOD) == 0ULL) reassign_blocks(&rt);
        if ((rt.step % REPORT_PERIOD)   == 0ULL) report_stats(&rt);

        rt.step++;
        if (DEMO_MAX_STEPS > 0ULL && rt.step >= DEMO_MAX_STEPS)
            rt.running = false;
    }

    printf("runtime finished at step=%llu\n", (unsigned long long)rt.step);
    return 0;
}
