# QUICK REFERENCE - MAJOR LAB TOOLS

## Available Tools

### 1. data_prep.exe — Data Cleaning

```bash
data_prep.exe <input.txt> <output.txt>
```

**What it does**: Deduplicates + filters low-quality lines
- FNV-1a hash for exact dedup
- Length filter: 3-2048 chars
- ASCII encoding check
- Entropy threshold: >= 0.1 bits

**Example**:
```bash
data_prep.exe raw_corpus.txt clean_corpus.txt
# Output: 8.8 KB → 5.4 KB (79/128 lines kept, 1.6x reduction)
```

**Output**: Audit trail showing Kept/Exact dups/Too short/Too long/Bad encoding/Low entropy

---

### 2. full_lm.exe — Language Model Engine

#### Training Commands:

```bash
# Train on single file
full_lm.exe train <corpus.txt> <model.bin> [passes] [slots] [ctx]

# Train on multiple files (list format)
full_lm.exe train-list <list.txt> <model.bin> [passes] [slots] [ctx]

# Resume training from checkpoint
full_lm.exe resume <model.bin> <corpus.txt> [passes]

# Resume training from multiple files
full_lm.exe resume-list <model.bin> <list.txt> [passes]
```

**Parameters**:
- `passes`: Training iterations over corpus (default 1)
- `slots`: Hash table slots per context (default 16)
- `ctx`: Context size = 2^ctx tokens (default 32768 = 15 bits)

#### Evaluation Commands:

```bash
# Evaluate on single file
full_lm.exe eval <model.bin> <test.txt> [online]

# Evaluate on multiple files
full_lm.exe eval-list <model.bin> <list.txt> [online]
```

**Parameters**:
- `online`: 1 = learn during eval, 0 = frozen weights (default 1 for train, 0 for test)

**Output**: Accuracy %, speed (Mtok/s), tokens processed

#### Generation Commands:

```bash
# Generate text from prompt
full_lm.exe gen <model.bin> <prompt> [max_length]

# Interactive chat
full_lm.exe chat <model.bin>
```

**Example**:
```bash
full_lm.exe gen model_best.bin "What is" 50
# Output: What is the context on hat is the context...

full_lm.exe chat model_best.bin
# Interactive: type prompts, type :quit to exit
```

**Features**:
- Confidence cascade (selects n-gram level by prediction confidence)
- Repetition detection (avoids "at at at...")
- Phrase-loop guards
- Context reset on newline

---

### 3. eval_metrics.exe — Rigorous Evaluation

```bash
# Single checkpoint evaluation
eval_metrics.exe <model.bin> <test.txt>

# Multi-checkpoint ranking
eval_metrics.exe <dummy> <test.txt> <checkpoint_list.txt>
```

**Output Metrics**:
- Perplexity: exp(cross_entropy), lower = better
- Cross-entropy: -sum(log(P(token)))/N
- Accuracy: %correct predictions
- Speed: Mtok/sec
- Composite score: 0.6×ppl_inv + 0.3×acc + 0.1×speed

**Example**:
```bash
eval_metrics.exe dummy holdout.txt checkpoint_list.txt
# Ranks all checkpoints by composite score
# Output: Best checkpoint: checkpoint_cycle_2.bin (score 0.689)
```

---

### 4. major_lab_infrastructure.ps1 — Complete Pipeline

Orchestrates all 6 phases end-to-end:

```powershell
.\major_lab_infrastructure.ps1 `
  -DataRoot . `
  -Pattern "*.txt" `
  -OutputDir major_lab_run `
  -Shards 4 `
  -Cycles 2 `
  -PassesPerCycle 1 `
  -Slots 16 `
  -CtxCount 32768
```

**Parameters**:
- `DataRoot`: Directory with corpus files (default: current dir)
- `Pattern`: File glob pattern (default: "*.txt")
- `OutputDir`: Output directory for artifacts
- `Shards`: Number of parallel training partitions
- `Cycles`: Number of training cycles (1+ = continuous improvement)
- `PassesPerCycle`: Training passes per cycle
- `Slots`: Hash table slots (higher = slower, more capacity)
- `CtxCount`: Context size (must be power of 2)

**Phases**:
1. Compile tools (if needed)
2. Data preparation (dedup + filter)
3. Corpus sharding (N equal partitions)
4. Distributed training (multi-cycle)
5. Perplexity evaluation (checkpoint ranking)
6. Inference showcase (sample generation)

---

## Workflow Examples

### Example 1: Quick Test Run

```powershell
# Use built-in corpus files
.\major_lab_infrastructure.ps1 -OutputDir test_run -Cycles 1 -Shards 2
```

**Time**: ~10 seconds
**Output**: 1 trained model in test_run/4_model_distributed.bin

### Example 2: Production Training (1MB Corpus)

```powershell
# Collect 1MB+ of quality text in data/ directory
.\major_lab_infrastructure.ps1 `
  -DataRoot data/ `
  -OutputDir production_run `
  -Shards 8 `
  -Cycles 5 `
  -PassesPerCycle 3 `
  -Slots 32
```

**Time**: ~5 minutes (CPU, 1MB corpus)
**Output**: 5 checkpoints, best one selected by perplexity

### Example 3: GPU-Distributed Training (Future)

```bash
# On machine 1 (shard 0,1)
full_lm.exe train-list shards_01.txt model_01.bin 10 32 32768

# On machine 2 (shard 2,3)  
full_lm.exe train-list shards_23.txt model_23.bin 10 32 32768

# Aggregate checkpoints (manual or via distributed tool)
eval_metrics.exe dummy test.txt checkpoint_list.txt
```

**Expected speedup**: 8x (4 shards × 2 machines)

---

## Files & Artifacts

### Outputs After Pipeline Run:

```
major_lab_run/
├── 1_raw_corpus.txt           — Raw input (8.8 KB)
├── 2_cleaned_corpus.txt        — After dedup (5.4 KB)
├── 3_shard_list.txt            — Shard file paths
├── 4_model_distributed.bin     — Final model
├── checkpoint_cycle_1.bin      — After cycle 1
├── checkpoint_cycle_2.bin      — After cycle 2 (best)
├── checkpoint_list.txt         — Checkpoint paths for eval
└── shards/
    ├── shard_0.txt            — Partition 0
    ├── shard_1.txt            — Partition 1
    ├── shard_2.txt            — Partition 2
    └── shard_3.txt            — Partition 3
```

### Model Format:

Binary format (no text overhead):
```
[Header: magic, version, slots, ctx_count]
[Unigram table: 256 × 4 bytes]
[Bigram table: 256×256 × 2 bytes]
[Trigram: tokens + counts]
[Fourgram: tokens + counts]
[Octagram: tokens + counts]
[Entangled: tokens + counts]
```

---

## Performance Targets

| Operation | Time | Size |
|-----------|------|------|
| Data prep (5 KB corpus) | 10 ms | ~6 MB mem |
| Train cycle (5 KB, 1 pass) | 50 ms | 6.4 MB model |
| Resume training | 30 ms | Same |
| Evaluate checkpoint | 100 ms | ~6 MB mem |
| Generate 100 tokens | 50 ms | Real-time |
| Full 2-cycle pipeline | 0.5 s | All above |

---

## Scaling Checklist

- [ ] **Phase 1 (Data)**: Collect 1MB+ corpus
  - [ ] Clean with data_prep.exe (target: 50% reduction)
  - [ ] Run pipeline with Shards=8, Cycles=10
  - [ ] Verify accuracy improves each cycle
  - [ ] Archive best checkpoint

- [ ] **Phase 2 (GPU)**: Add GPU acceleration
  - [ ] Port training loop to CUDA
  - [ ] Parallelize table updates
  - [ ] Test on 4 GPUs = 100x speedup
  - [ ] Benchmark perplexity

- [ ] **Phase 3 (Distributed)**: Deploy to cluster
  - [ ] Containerize tools
  - [ ] Setup Kubernetes/Ray
  - [ ] Aggregate gradients across machines
  - [ ] Test on 10 machines

- [ ] **Phase 4 (Production)**: Deploy inference
  - [ ] Build API server (FastAPI)
  - [ ] Implement batching
  - [ ] Deploy to cloud (AWS/GCP)
  - [ ] Monitor metrics

---

## Troubleshooting

### "Failed to train from list"
- Check shard list file format (one path per line)
- Verify all shard files exist
- Run `full_lm.exe train` on single file first to test

### "Kept: 0" from data_prep
- Entropy threshold too high (fixed in latest build)
- Try lowering MIN_ENTROPY_BITS in data_prep.c
- Recompile: `gcc -O2 -o data_prep.exe data_prep.c`

### Model file not found
- Run training first: `full_lm.exe train-list`
- Check output directory has 4_model_distributed.bin
- Verify file size > 1 MB (model data)

### Generation produces garbage
- Check model was trained on enough data (>100 lines recommended)
- Try generation after eval to learn during eval
- Seed with more context: longer prompt = better output

---

## Next Steps

1. **Expand corpus** to 1MB (200x current)
   - Download Wikipedia, papers, code
   - Run data_prep for dedup
   - Measure accuracy improvement

2. **Add GPU support** (100x speedup)
   - Parallelize table updates with CUDA
   - Test on single GPU first
   - Scale to 4+ GPUs

3. **Deploy inference API**
   - FastAPI server wrapping full_lm.exe
   - Batch processing for throughput
   - Monitor perplexity + latency

4. **Continuous training**
   - Automated data collection pipeline
   - Daily retraining on new data
   - Auto-checkpoint selection + deploy

---

## References

- **FNV-1a hash**: Fast deduplication in O(1) time
- **Perplexity**: Standard LM evaluation metric (lower = better)
- **Fused architecture**: Combines 4 n-gram tables with XOR accumulator
- **Streaming I/O**: Unlimited corpus size via 1MB chunks
