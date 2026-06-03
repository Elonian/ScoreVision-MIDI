# ScoreVision MIDI CRNN/CNNT Evaluation

This is the single combined evaluation report for the completed CRNN and CNNT runs.

## Dataset And Protocol

| Item | Value |
| --- | --- |
| Local dataset | `data/grandstaff_dataset` |
| Train split | 41,598 samples |
| Validation split | 4,623 samples |
| Test split | 7,661 samples |
| Encoding used here | `.bekrn` / BEKRN-style tokens |
| Metrics | CER, SER, LER using the same parser as `works/ijdar-e2e-pianoform/eval_functions.py` |
| Fresh test protocol | Reload `best.pt`, run checkpoint inference on the full test split, `batch_size=1`, write `hyp/<index>.krn` and `gt/<index>.krn` |

## About The Screenshots

The screenshots in `outputs/Screenshot 2026-06-02 at 10.33.01 AM.png` and `outputs/Screenshot 2026-06-02 at 10.33.05 AM.png` are **not the same experiment** as the CRNN/CNNT GrandStaff run evaluated here.

| Screenshot table | What it is | Why it is not directly comparable to this run |
| --- | --- | --- |
| Table 1 | Data-source summary for `FP-GrandStaff`, `Mozarteum`, and `Polish Digital Scores` | Our run uses local `data/grandstaff_dataset` with 41,598/4,623/7,661 train/val/test samples. The screenshot table says `FP-GrandStaff` has 688 pages. |
| Table 2 | Test-set CER/SER/LER for `FP-GRANDSTAFF` using `SMT_CNN` and `SMT_NeXt` with KERN/EKERN/BEKERN encodings | Our run evaluates CRNN and CNNT checkpoints trained on local GrandStaff `.bekrn`; it is a different model family and dataset split. |

So the screenshot numbers, for example BEKERN `SMT_NeXt` CER/SER/LER of `5.6 / 6.9 / 12.9`, are useful context but not an apples-to-apples target for this local CRNN/CNNT evaluation.

## Fresh Full-Test Checkpoint Results

| Model | Checkpoint | Checkpoint epoch | Test samples | CER | SER | LER | Hyp files | GT files | Metrics JSON |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CRNN | `outputs/scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed/weights/best.pt` | 81 | 7,661 | 4.3678 | 7.2036 | 19.1032 | 7,661 | 7,661 | `evaluation/runs/crnn_test/metrics.json` |
| CNNT | `outputs/scorevision_grandstaff_bekrn_cnnt_ddp_b2/weights/best.pt` | 70 | 7,661 | 3.9117 | 6.3623 | 16.3517 | 7,661 | 7,661 | `evaluation/runs/cnnt_test/metrics.json` |

CNNT is better than CRNN on all three fresh checkpoint metrics for this local GrandStaff test split.

## Training-Loop Test Values

These are the earlier values written by the training loop at the end of training. They differ slightly from the fresh checkpoint run above because that path used the configured evaluation batches and padding, while the fresh run uses paper-style `batch_size=1`.

| Model | Test CER | Test SER | Test LER | Source log |
| --- | ---: | ---: | ---: | --- |
| CRNN | 4.2383 | 7.0236 | 18.6882 | `logs/scorevision_grandstaff_bekrn_crnn_ddp_b12_bucketed_20260530_203444.log` |
| CNNT | 3.8116 | 6.2293 | 16.0543 | `logs/scorevision_grandstaff_bekrn_cnnt_ddp_b2_20260530_030814.log` |

## Every-Epoch Validation Progression

This is the epoch-by-epoch evaluation used during training and early stopping: validation CER/SER/LER after each epoch. The paper-style full test is normally run once on the selected best checkpoint, not on every epoch checkpoint. All listed epoch checkpoints exist locally.

| Model | Epoch | Train loss | Val CER | Val SER | Val LER | New best Val SER | Selected best checkpoint | Epoch checkpoint exists |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| CRNN | 1 | 2.903102 | 35.4715 | 51.2663 | 96.2817 | yes |  | yes |
| CRNN | 2 | 2.010211 | 29.1802 | 43.6733 | 81.3530 | yes |  | yes |
| CRNN | 3 | 1.285797 | 24.0207 | 37.0162 | 74.2203 | yes |  | yes |
| CRNN | 4 | 0.977224 | 21.3709 | 33.1304 | 69.9420 | yes |  | yes |
| CRNN | 5 | 0.813273 | 17.9253 | 27.9123 | 61.4415 | yes |  | yes |
| CRNN | 6 | 0.674078 | 15.4367 | 23.9457 | 53.4967 | yes |  | yes |
| CRNN | 7 | 0.559121 | 13.4270 | 20.9597 | 47.4208 | yes |  | yes |
| CRNN | 8 | 0.474458 | 11.9269 | 18.6232 | 42.7428 | yes |  | yes |
| CRNN | 9 | 0.414862 | 10.9287 | 17.0671 | 39.4296 | yes |  | yes |
| CRNN | 10 | 0.374879 | 9.9266 | 15.5549 | 36.2507 | yes |  | yes |
| CRNN | 11 | 0.342245 | 9.5754 | 14.9083 | 34.7018 | yes |  | yes |
| CRNN | 12 | 0.316381 | 8.8222 | 13.8376 | 32.5482 | yes |  | yes |
| CRNN | 13 | 0.292371 | 8.7522 | 13.7538 | 32.7028 | yes |  | yes |
| CRNN | 14 | 0.275091 | 7.9467 | 12.5390 | 29.8283 | yes |  | yes |
| CRNN | 15 | 0.256130 | 7.6855 | 11.9934 | 28.5670 | yes |  | yes |
| CRNN | 16 | 0.243826 | 7.5073 | 11.7147 | 27.8941 | yes |  | yes |
| CRNN | 17 | 0.229321 | 7.1169 | 11.1815 | 26.8841 | yes |  | yes |
| CRNN | 18 | 0.216773 | 6.8797 | 10.8496 | 26.1833 | yes |  | yes |
| CRNN | 19 | 0.206947 | 6.6556 | 10.4846 | 25.3338 | yes |  | yes |
| CRNN | 20 | 0.199095 | 6.5459 | 10.2948 | 24.8555 | yes |  | yes |
| CRNN | 21 | 0.191216 | 6.4629 | 10.1151 | 24.4415 | yes |  | yes |
| CRNN | 22 | 0.183785 | 6.1568 | 9.7159 | 23.4593 | yes |  | yes |
| CRNN | 23 | 0.176407 | 5.9543 | 9.3907 | 22.6759 | yes |  | yes |
| CRNN | 24 | 0.169973 | 5.8797 | 9.2671 | 22.3823 | yes |  | yes |
| CRNN | 25 | 0.171754 | 5.7986 | 9.1718 | 22.2677 | yes |  | yes |
| CRNN | 26 | 0.161671 | 5.8831 | 9.2196 | 22.2542 |  |  | yes |
| CRNN | 27 | 0.155653 | 5.5238 | 8.7020 | 21.0761 | yes |  | yes |
| CRNN | 28 | 0.151522 | 5.5872 | 8.7968 | 21.3881 |  |  | yes |
| CRNN | 29 | 0.149764 | 5.6605 | 8.8811 | 21.4730 |  |  | yes |
| CRNN | 30 | 0.147457 | 5.3354 | 8.4261 | 20.5210 | yes |  | yes |
| CRNN | 31 | 0.142527 | 5.1575 | 8.1700 | 19.8566 | yes |  | yes |
| CRNN | 32 | 0.140229 | 5.2676 | 8.3039 | 20.1883 |  |  | yes |
| CRNN | 33 | 0.145362 | 5.2419 | 8.2880 | 20.1384 |  |  | yes |
| CRNN | 34 | 0.142941 | 5.2193 | 8.2558 | 20.0733 |  |  | yes |
| CRNN | 35 | 0.134795 | 4.9514 | 7.8408 | 19.0354 | yes |  | yes |
| CRNN | 36 | 0.131114 | 4.9806 | 7.8739 | 19.1743 |  |  | yes |
| CRNN | 37 | 0.127171 | 4.9689 | 7.8713 | 19.1594 |  |  | yes |
| CRNN | 38 | 0.125248 | 4.7914 | 7.6075 | 18.5967 | yes |  | yes |
| CRNN | 39 | 0.120720 | 4.7398 | 7.5033 | 18.2708 | yes |  | yes |
| CRNN | 40 | 0.120102 | 4.8283 | 7.6444 | 18.5639 |  |  | yes |
| CRNN | 41 | 0.121356 | 4.6946 | 7.4848 | 18.2110 | yes |  | yes |
| CRNN | 42 | 0.118010 | 4.7238 | 7.4767 | 18.1391 | yes |  | yes |
| CRNN | 43 | 0.118371 | 4.7354 | 7.4935 | 18.1436 |  |  | yes |
| CRNN | 44 | 0.114920 | 4.5843 | 7.2547 | 17.5291 | yes |  | yes |
| CRNN | 45 | 0.113762 | 4.5387 | 7.2015 | 17.3866 | yes |  | yes |
| CRNN | 46 | 0.110891 | 4.4654 | 7.0962 | 17.2064 | yes |  | yes |
| CRNN | 47 | 0.109700 | 4.5744 | 7.2390 | 17.5570 |  |  | yes |
| CRNN | 48 | 0.106480 | 4.3477 | 6.9018 | 16.7133 | yes |  | yes |
| CRNN | 49 | 0.103347 | 4.4715 | 7.0821 | 17.1533 |  |  | yes |
| CRNN | 50 | 0.103380 | 4.3624 | 6.8927 | 16.5928 | yes |  | yes |
| CRNN | 51 | 0.107124 | 4.4490 | 7.0218 | 16.9848 |  |  | yes |
| CRNN | 52 | 0.102709 | 4.7194 | 7.4136 | 18.0159 |  |  | yes |
| CRNN | 53 | 0.102015 | 4.2574 | 6.7411 | 16.3577 | yes |  | yes |
| CRNN | 54 | 0.102952 | 4.1559 | 6.5934 | 15.9738 | yes |  | yes |
| CRNN | 55 | 0.098202 | 4.3766 | 6.8946 | 16.6135 |  |  | yes |
| CRNN | 56 | 0.100552 | 4.1696 | 6.6466 | 16.1604 |  |  | yes |
| CRNN | 57 | 0.095590 | 4.0550 | 6.4651 | 15.7131 | yes |  | yes |
| CRNN | 58 | 0.096029 | 4.1126 | 6.5568 | 15.9181 |  |  | yes |
| CRNN | 59 | 0.092538 | 4.2972 | 6.7949 | 16.2710 |  |  | yes |
| CRNN | 60 | 0.097383 | 4.1069 | 6.5002 | 15.6390 |  |  | yes |
| CRNN | 61 | 0.098754 | 4.0361 | 6.4318 | 15.5563 | yes |  | yes |
| CRNN | 62 | 0.091606 | 4.0081 | 6.3778 | 15.4646 | yes |  | yes |
| CRNN | 63 | 0.091385 | 3.9712 | 6.3307 | 15.2803 | yes |  | yes |
| CRNN | 64 | 0.088924 | 4.0156 | 6.3755 | 15.3827 |  |  | yes |
| CRNN | 65 | 0.092512 | 4.0241 | 6.3736 | 15.3877 |  |  | yes |
| CRNN | 66 | 0.090318 | 4.1122 | 6.5026 | 15.7370 |  |  | yes |
| CRNN | 67 | 0.089913 | 3.8884 | 6.1899 | 14.9683 | yes |  | yes |
| CRNN | 68 | 0.089358 | 4.0146 | 6.3892 | 15.4255 |  |  | yes |
| CRNN | 69 | 0.089820 | 3.8326 | 6.0988 | 14.7543 | yes |  | yes |
| CRNN | 70 | 0.087036 | 4.0195 | 6.3753 | 15.4834 |  |  | yes |
| CRNN | 71 | 0.086806 | 3.8066 | 6.0656 | 14.7485 | yes |  | yes |
| CRNN | 72 | 0.087804 | 3.7440 | 5.9663 | 14.4509 | yes |  | yes |
| CRNN | 73 | 0.085828 | 3.8059 | 6.0660 | 14.7022 |  |  | yes |
| CRNN | 74 | 0.083785 | 3.8077 | 6.0616 | 14.6703 |  |  | yes |
| CRNN | 75 | 0.086307 | 3.7178 | 5.9613 | 14.5287 | yes |  | yes |
| CRNN | 76 | 0.085134 | 3.7098 | 5.9308 | 14.3772 | yes |  | yes |
| CRNN | 77 | 0.087615 | 3.6931 | 5.9049 | 14.3925 | yes |  | yes |
| CRNN | 78 | 0.086312 | 3.7956 | 6.0287 | 14.6033 |  |  | yes |
| CRNN | 79 | 0.093818 | 4.0373 | 6.3629 | 15.4565 |  |  | yes |
| CRNN | 80 | 0.094158 | 3.6737 | 5.8570 | 14.2541 | yes |  | yes |
| CRNN | 81 | 0.082886 | 3.6329 | 5.8109 | 14.1089 | yes | yes | yes |
| CRNN | 82 | 0.083052 | 3.7320 | 5.9349 | 14.4280 |  |  | yes |
| CRNN | 83 | 0.081700 | 3.7772 | 6.0117 | 14.5966 |  |  | yes |
| CRNN | 84 | 0.084502 | 3.8613 | 6.1344 | 14.9670 |  |  | yes |
| CRNN | 85 | 0.081821 | 3.7792 | 6.0201 | 14.6627 |  |  | yes |
| CRNN | 86 | 0.085350 | 3.6622 | 5.8376 | 14.1588 |  |  | yes |
| CNNT | 1 | 2.602427 | 35.0541 | 51.5837 | 93.8414 | yes |  | yes |
| CNNT | 2 | 2.144253 | 31.8784 | 48.0640 | 85.8381 | yes |  | yes |
| CNNT | 3 | 1.657253 | 28.3744 | 43.8833 | 81.0091 | yes |  | yes |
| CNNT | 4 | 1.275288 | 24.6299 | 38.8812 | 75.5046 | yes |  | yes |
| CNNT | 5 | 1.037055 | 23.3694 | 37.2043 | 76.7483 | yes |  | yes |
| CNNT | 6 | 0.856443 | 20.0918 | 31.9477 | 68.8367 | yes |  | yes |
| CNNT | 7 | 0.716653 | 17.3476 | 27.6538 | 61.7382 | yes |  | yes |
| CNNT | 8 | 0.609684 | 14.7637 | 23.4936 | 53.8913 | yes |  | yes |
| CNNT | 9 | 0.527817 | 12.9903 | 20.7468 | 48.5746 | yes |  | yes |
| CNNT | 10 | 0.457570 | 11.5198 | 18.3211 | 43.4737 | yes |  | yes |
| CNNT | 11 | 0.407176 | 10.5387 | 16.6416 | 39.5231 | yes |  | yes |
| CNNT | 12 | 0.368745 | 9.7694 | 15.3102 | 36.1626 | yes |  | yes |
| CNNT | 13 | 0.333631 | 8.9986 | 14.0976 | 33.5794 | yes |  | yes |
| CNNT | 14 | 0.307341 | 8.3792 | 13.1140 | 31.2154 | yes |  | yes |
| CNNT | 15 | 0.283701 | 8.0822 | 12.6095 | 30.2189 | yes |  | yes |
| CNNT | 16 | 0.262751 | 7.6298 | 11.9475 | 28.9486 | yes |  | yes |
| CNNT | 17 | 0.245719 | 7.3893 | 11.4774 | 27.6756 | yes |  | yes |
| CNNT | 18 | 0.232652 | 7.0136 | 10.8887 | 25.9132 | yes |  | yes |
| CNNT | 19 | 0.220250 | 6.5765 | 10.2066 | 24.1907 | yes |  | yes |
| CNNT | 20 | 0.209224 | 6.5003 | 10.0625 | 23.9142 | yes |  | yes |
| CNNT | 21 | 0.202137 | 6.0766 | 9.3811 | 21.9778 | yes |  | yes |
| CNNT | 22 | 0.197199 | 5.9852 | 9.2065 | 21.5225 | yes |  | yes |
| CNNT | 23 | 0.187386 | 5.6268 | 8.6919 | 20.1209 | yes |  | yes |
| CNNT | 24 | 0.181239 | 5.5783 | 8.5886 | 19.9497 | yes |  | yes |
| CNNT | 25 | 0.173824 | 5.4792 | 8.4118 | 19.6391 | yes |  | yes |
| CNNT | 26 | 0.165448 | 5.1780 | 7.9831 | 18.4713 | yes |  | yes |
| CNNT | 27 | 0.161159 | 5.1316 | 7.8550 | 18.0919 | yes |  | yes |
| CNNT | 28 | 0.155046 | 5.0252 | 7.7112 | 17.6878 | yes |  | yes |
| CNNT | 29 | 0.150196 | 4.9609 | 7.6217 | 17.4383 | yes |  | yes |
| CNNT | 30 | 0.148041 | 4.8955 | 7.5167 | 17.2617 | yes |  | yes |
| CNNT | 31 | 0.141383 | 4.7381 | 7.3431 | 17.0481 | yes |  | yes |
| CNNT | 32 | 0.139784 | 4.9460 | 7.5603 | 17.3277 |  |  | yes |
| CNNT | 33 | 0.137810 | 4.6371 | 7.1517 | 16.4989 | yes |  | yes |
| CNNT | 34 | 0.138282 | 4.4902 | 6.9447 | 15.9657 | yes |  | yes |
| CNNT | 35 | 0.130605 | 4.5927 | 7.0861 | 16.2539 |  |  | yes |
| CNNT | 36 | 0.130129 | 4.4588 | 6.8978 | 15.9581 | yes |  | yes |
| CNNT | 37 | 0.125195 | 4.4658 | 6.8953 | 15.8507 | yes |  | yes |
| CNNT | 38 | 0.124649 | 4.4234 | 6.8359 | 15.7374 | yes |  | yes |
| CNNT | 39 | 0.123503 | 4.2990 | 6.6676 | 15.4407 | yes |  | yes |
| CNNT | 40 | 0.117425 | 4.2239 | 6.5317 | 14.9341 | yes |  | yes |
| CNNT | 41 | 0.115402 | 4.2620 | 6.5683 | 15.1225 |  |  | yes |
| CNNT | 42 | 0.114959 | 4.1103 | 6.3578 | 14.5485 | yes |  | yes |
| CNNT | 43 | 0.110117 | 4.1268 | 6.3420 | 14.4352 | yes |  | yes |
| CNNT | 44 | 0.111838 | 4.0904 | 6.3553 | 14.5763 |  |  | yes |
| CNNT | 45 | 0.108224 | 4.0640 | 6.2959 | 14.3552 | yes |  | yes |
| CNNT | 46 | 0.107750 | 4.3100 | 6.5742 | 14.8110 |  |  | yes |
| CNNT | 47 | 0.117343 | 4.1919 | 6.4315 | 14.7238 |  |  | yes |
| CNNT | 48 | 0.108101 | 3.9805 | 6.1490 | 13.9862 | yes |  | yes |
| CNNT | 49 | 0.104755 | 3.8771 | 6.0292 | 13.7479 | yes |  | yes |
| CNNT | 50 | 0.101500 | 3.8127 | 5.9544 | 13.6288 | yes |  | yes |
| CNNT | 51 | 0.099350 | 3.8315 | 5.9564 | 13.5286 |  |  | yes |
| CNNT | 52 | 0.099291 | 3.8854 | 6.0054 | 13.5857 |  |  | yes |
| CNNT | 53 | 0.099177 | 3.8388 | 5.9084 | 13.3155 | yes |  | yes |
| CNNT | 54 | 0.102657 | 3.7402 | 5.8346 | 13.1919 | yes |  | yes |
| CNNT | 55 | 0.099372 | 3.7292 | 5.7729 | 12.9721 | yes |  | yes |
| CNNT | 56 | 0.095486 | 3.6355 | 5.6769 | 12.8548 | yes |  | yes |
| CNNT | 57 | 0.094905 | 3.9324 | 6.0986 | 13.8774 |  |  | yes |
| CNNT | 58 | 0.094723 | 3.6341 | 5.6749 | 12.8777 | yes |  | yes |
| CNNT | 59 | 0.094353 | 3.5712 | 5.5818 | 12.6610 | yes |  | yes |
| CNNT | 60 | 0.091434 | 3.6705 | 5.6859 | 12.8255 |  |  | yes |
| CNNT | 61 | 0.091645 | 3.6181 | 5.6623 | 12.9878 |  |  | yes |
| CNNT | 62 | 0.091802 | 3.6549 | 5.6954 | 13.0314 |  |  | yes |
| CNNT | 63 | 0.090942 | 3.6929 | 5.7034 | 12.9011 |  |  | yes |
| CNNT | 64 | 0.088880 | 3.5556 | 5.5107 | 12.4862 | yes |  | yes |
| CNNT | 65 | 0.088974 | 3.8882 | 5.9834 | 13.5591 |  |  | yes |
| CNNT | 66 | 0.092452 | 3.5405 | 5.5129 | 12.4597 |  |  | yes |
| CNNT | 67 | 0.087536 | 3.4894 | 5.4625 | 12.4475 | yes |  | yes |
| CNNT | 68 | 0.087420 | 3.5489 | 5.5293 | 12.5253 |  |  | yes |
| CNNT | 69 | 0.087915 | 3.4647 | 5.4154 | 12.2443 | yes |  | yes |
| CNNT | 70 | 0.086139 | 3.2916 | 5.1627 | 11.6290 | yes | yes | yes |
| CNNT | 71 | 0.084275 | 3.3807 | 5.2950 | 11.9787 |  |  | yes |
| CNNT | 72 | 0.084161 | 3.4235 | 5.3586 | 12.1738 |  |  | yes |
| CNNT | 73 | 0.085918 | 3.3532 | 5.2459 | 11.8389 |  |  | yes |
| CNNT | 74 | 0.083454 | 3.3715 | 5.2965 | 12.0421 |  |  | yes |
| CNNT | 75 | 0.086736 | 3.3114 | 5.2014 | 11.7589 |  |  | yes |

## Regeneration Commands

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
  --device cuda:1
```
