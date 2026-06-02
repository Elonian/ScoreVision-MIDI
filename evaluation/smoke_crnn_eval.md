# ScoreVision MIDI CTC Evaluation

Dataset: GrandStaff `test.txt` unless the split column says otherwise. The metrics are computed with the same CER/SER/LER parser used by the official IJDAR code. Checkpoint mode uses the paper-style test batch size of 1 by default, which writes `hyp/0.krn`, `hyp/1.krn`, ... and matching `gt` files.

| Model | Source | Split | Samples | CER | SER | LER | Checkpoint epoch | Hyp dir | GT dir |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CRNN | checkpoint_inference | test | 2 | 3.4216 | 5.3419 | 19.1489 | 81 | `evaluation/runs/smoke_crnn/hyp` | `evaluation/runs/smoke_crnn/gt` |

## Metric Definitions

- `CER`: Character Error Rate: normalized Levenshtein distance after character-level parsing.
- `SER`: Symbol Error Rate: normalized Levenshtein distance after token/symbol parsing.
- `LER`: Line Error Rate: normalized Levenshtein distance after line-level parsing.
