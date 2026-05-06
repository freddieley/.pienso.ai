# Contributing

## Workflow
1. Create a feature branch from `main`.
2. Keep commits focused and small.
3. Run local compile and a short train/eval smoke test before opening a PR.
4. Include command output snippets in PR description when relevant.

## Coding Rules
- Keep `main.c` unchanged unless explicitly required.
- Prefer streaming I/O for large corpora.
- Preserve integer/bitwise-first logic in model hot paths.
- Avoid introducing floating point in prediction/sampling critical loops.

## Minimum Validation
```powershell
gcc -O2 -o full_lm.exe full_lm.c
gcc -O2 -o word_lm.exe word_lm.c
gcc -O2 -o data_prep.exe data_prep.c

.\full_lm.exe train conversation_train.txt model_smoke.bin 1
.\full_lm.exe eval model_smoke.bin holdout_eval.txt 0
```
