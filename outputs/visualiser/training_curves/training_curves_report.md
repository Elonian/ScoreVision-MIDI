# Training Curve Summary

This report explains the CRNN/CNNT training curves generated from the real training logs.

## Outputs

| File | Purpose |
| --- | --- |
| `training_dashboard.png` / `.svg` | Four-panel presentation dashboard. |
| `loss_vs_validation_ser.png` / `.svg` | Focused plot linking loss to the early-stopping metric. |
| `validation_metrics_comparison.png` / `.svg` | CER/SER/LER comparison by epoch. |
| `epoch_progression.csv` | Every-epoch numeric values for CRNN and CNNT. |
| `curve_summary.json` | Best epochs, final metrics, and fresh test metrics. |

## Key Reading

| Model | Epochs | Best epoch | Best Val SER | Final Val SER | Fresh Test SER |
| --- | ---: | ---: | ---: | ---: | ---: |
| CRNN | 86 | 81 | 5.8109 | 5.8376 | 7.2036 |
| CNNT | 75 | 70 | 5.1627 | 5.2014 | 6.3623 |

CNNT has the better validation and fresh full-test SER. The loss curves keep moving late in training, but the validation curves show the useful plateau and justify early stopping.

## Theory Notes For Presentation

- Training loss measures how well the model fits batches seen during optimization.
- Validation CER/SER/LER measure recognition quality on held-out pages.
- SER is the early-stopping metric in these runs, so the selected checkpoint is the epoch with the best validation SER before patience runs out.
- The gap between CRNN and CNNT is clearest on SER/LER: CNNT's transformer unfolding decoder handles long page-level dependencies better than the recurrent decoder in this run.
- The fresh full-test numbers are computed after reloading `best.pt` and evaluating the full test split with `batch_size=1`.
