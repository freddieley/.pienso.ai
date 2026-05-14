# pienso.ai Language Model Lab

CPU-first C language model lab focused on integer/bitwise methods, streaming data pipelines, and scalable corpus training.

## Highlights
- Byte-level model: `full_lm.c` (fused n-gram + XOR context, no float hot path)
- Word-level model: `word_lm.c` (word n-grams with open-addressing vocab)
- Corpus cleaner: `data_prep.c`
- Distributed shard training: `scale_train.ps1`
- Large corpus builder (5B+ byte-token target): `build_corpus.ps1`

## Repository Layout
- `full_lm.c` - byte-level LM runtime and CLI
- `word_lm.c` - word-level LM runtime and CLI
- `data_prep.c` - corpus dedup/filter utility
- `eval_metrics.c` - evaluation helper
- `build_corpus.ps1` - Gutenberg + Wikipedia corpus builder
- `scale_train.ps1` - parallel shard train/merge/eval orchestration
- `docs/` - project notes and infrastructure docs
- `legacy/` - archived experimental source files

## Prerequisites
- Windows PowerShell 5.1+
- `gcc` on PATH (MinGW recommended)

## Build
```powershell
gcc -O2 -o full_lm.exe full_lm.c
gcc -O2 -o word_lm.exe word_lm.c
gcc -O2 -o data_prep.exe data_prep.c
gcc -O2 -o eval_metrics.exe eval_metrics.c
```

## Quick Start
### 1) Build a corpus
```powershell
.\build_corpus.ps1 -GutenbergTarget 10000 -WikiBatches 5000 -Parallel 8 -Resume
```

### 2) Train byte-level model
```powershell
.\scale_train.ps1 -Corpus big_corpus.txt -OutDir scale_run -Shards 16 -Cycles 3
```

### 3) Train word-level model
```powershell
.\word_lm.exe train big_corpus.txt word_model.wlm 4
```

### 4) Evaluate or generate
```powershell
.\full_lm.exe eval scale_run\model_merged.bin holdout_eval.txt 0
.\full_lm.exe gen scale_run\model_merged.bin "hello" 256
.\word_lm.exe eval word_model.wlm holdout_eval.txt
.\word_lm.exe gen-sample word_model.wlm "machine learning" 64 1 123
```

### 5) Fast adaptation mode (instant correction)
```powershell
.\\full_lm.exe patch-add quick_patches.txt "machine learning" " is solved by fast patch memory."
.\\full_lm.exe patch-stats quick_patches.txt
.\\full_lm.exe gen-fast scale_run\\model_merged.bin quick_patches.txt "machine learning" 80 220 ?
```

### 6) Nested learning experiment
```powershell
.\\full_lm.exe eval-nested tiny_nested.bin tiny_test.txt 256
.\\full_lm.exe gen-nested tiny_nested.bin "nested learning" 80 123
```
Nested mode keeps the base model frozen and lets a fast inner learner adapt online within a stream or file window. That makes it a good fit for testing rapid adaptation with very little data.

## Notes on Artifacts
Model binaries, checkpoints, corpora, and executables are intentionally git-ignored. Keep source, scripts, and docs in Git; keep large artifacts local or publish them as release assets.

## GitHub Preparation
- `.gitignore`, `.gitattributes`, and `.editorconfig` are included
- See `docs/REPO_SETUP.md` for first-push checklist
- Use `CONTRIBUTING.md` for contribution workflow

## License
MIT (see `LICENSE`).
