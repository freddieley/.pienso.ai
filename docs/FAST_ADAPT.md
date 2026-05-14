# Fast Adaptation Mode

This project now includes a fast adaptation path in full_lm.exe that prioritizes exact correctness on known prompts and abstains when confidence is too low.

## New Commands

- patch-add <patch.txt> <prompt> <response>
- patch-stats <patch.txt>
- gen-fast <model.bin> <patch.txt> <prompt> [max_len=256] [min_conf=160] [abstain=?]

## Behavior

1. gen-fast checks patch memory first.
2. If prompt match exists, response is returned immediately.
3. If no patch match exists, model generation runs with confidence gate.
4. If confidence drops below min_conf, output abstain token and stop.

## One-shot correction loop

1. Run query with gen-fast.
2. If output is wrong, add correction with patch-add.
3. Repeat query; correction applies immediately.

## Example

full_lm.exe patch-add quick_patches.txt "machine learning" " is solved by fast patch memory."
full_lm.exe gen-fast smoke.bin quick_patches.txt "machine learning" 80 220 ?

## Batch evaluation

Use eval_fast_adapt.ps1 with a tab-separated file prompt<TAB>expected_suffix.

## Nested learning experiment

The nested-learning path in full_lm.exe uses a frozen outer model plus a fast online inner learner.

Commands:

- eval-nested <model.bin> <test.txt> [window=4096]
- eval-nested-list <model.bin> <files.txt> [window=4096]
- gen-nested <model.bin> <prompt> [max_len=256] [seed=0]

Observed smoke-test result on a tiny stream:

- frozen base accuracy: 13.11%
- nested online-inner accuracy: 61.48%
