# ScoreVision MIDI CRNN/CNNT Evaluation

Dataset: GrandStaff test split, `data/grandstaff_dataset/partitions/test.txt`.

Samples: 7,661.

Metrics match the official IJDAR evaluation code in `works/ijdar-e2e-pianoform/eval_functions.py`:

- `CER`: Character Error Rate, normalized Levenshtein distance after character-level `.krn` parsing.
- `SER`: Symbol Error Rate, normalized Levenshtein distance after symbol/token-level `.krn` parsing.
- `LER`: Line Error Rate, normalized Levenshtein distance after line-level `.krn` parsing.

## Full Test Results

| Model | Checkpoint | Best epoch | Test samples | CER | SER | LER | Hyp files | GT files |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CRNN | `outputs/scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed/weights/best.pt` | 81 | 7,661 | 4.2383 | 7.0236 | 18.6882 | 7,661 | 7,661 |
| CNNT | `outputs/scorevision_grandstaff_bekrn_cnnt_ddp_b2/weights/best.pt` | 70 | 7,661 | 3.8116 | 6.2293 | 16.0543 | 7,661 | 7,661 |

## Source Logs

| Model | Log file | Final epoch | Early stopping |
| --- | --- | ---: | --- |
| CRNN | `logs/scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed_20260530_203444.log` | 86 | `best_epoch=81 best_metric=5.810880` |
| CNNT | `logs/scorevision_grandstaff_bekrn_cnnt_ddp_b2_20260530_030814.log` | 75 | `best_epoch=70 best_metric=5.162688` |

## Regeneration Commands

These commands rerun inference from `best.pt` and write paper-style prediction files: `hyp/0.krn`, `hyp/1.krn`, ... with matching `gt` files.

```bash
python evaluation/evaluate_ctc_checkpoint.py \
  --config configs/score_unfolding.yaml \
  --checkpoint outputs/scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed/weights/best.pt \
  --output-dir evaluation/runs/crnn_test \
  --split test \
  --device cuda:0

python evaluation/evaluate_ctc_checkpoint.py \
  --config configs/score_unfolding_cnnt.yaml \
  --checkpoint outputs/scorevision_grandstaff_bekrn_cnnt_ddp_b2/weights/best.pt \
  --output-dir evaluation/runs/cnnt_test \
  --split test \
  --device cuda:0
```

For quick validation, the smoke tests with `--max-samples 2` completed successfully for both models.
