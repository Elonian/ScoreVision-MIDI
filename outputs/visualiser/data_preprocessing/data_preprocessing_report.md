# GrandStaff Data Preprocessing and Statistics

This report was generated from the local GrandStaff dataset used by the CRNN and CNNT runs.

## Data Source

- Data root: `data`
- Partitions: `data/grandstaff_dataset/partitions/train.txt`, `val.txt`, `test.txt`
- Training image cache: `outputs/image_cache/grandstaff_bekrn_plain`
- Total partition entries: `53,882`
- Cached model-input images found: `53,882`

## Preprocessing Pipeline

1. Read each partition file and resolve the relative path to a `.bekrn` transcription.
2. Convert BEKRN text to CTC target tokens with the same tokenizer used by training.
3. Encode spaces as `<s>`, tabs as `<t>`, and line breaks as `<b>`.
4. Read the clean `.jpg` score image as grayscale. The reported CRNN/CNNT configs set `load_distorted: false`.
5. Resize with ratio `1.0`, rotate the image 90 degrees clockwise, and store it as `.npy` cache.
6. During training/evaluation, convert to float tensor with `image / 255.0`, pad each batch, and use CTC loss/decoding.
7. CTC input length is `(cached_width // 8) * (cached_height // 16)`.

## Split Statistics

| Split | Samples | Clean JPG | Distorted JPG Present | Cached NPY | Vocab | Token median | Token p95 | Token max | Height median | Width median | Min CTC margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 41,598 | 41,598 | 41,598 | 41,598 | 185 | 333 | 684 | 1,716 | 723 | 256 | 195 |
| Validation | 4,623 | 4,623 | 4,623 | 4,623 | 170 | 335 | 683.80 | 1,676 | 730 | 256 | 244 |
| Test | 7,661 | 7,661 | 7,661 | 7,661 | 179 | 338 | 672 | 1,412 | 726 | 256 | 248 |
| Total | 53,882 | 53,882 | 53,882 | 53,882 | 187 | 334 | 682 | 1,716 | 724 | 256 | 195 |

## Composer Coverage

| Composer | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| beethoven | 19,123 | 2,130 | 3,467 | 24,720 |
| scarlatti-d | 6,875 | 776 | 1,286 | 8,937 |
| mozart | 6,249 | 655 | 1,144 | 8,048 |
| chopin | 4,995 | 585 | 955 | 6,535 |
| joplin | 4,187 | 459 | 778 | 5,424 |
| hummel | 169 | 18 | 31 | 218 |

## Top 25 Tokens

| Rank | Token | Count | Percent | Train | Validation | Test |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `<t>` | 3,198,544 | 16.15% | 2,462,946 | 274,506 | 461,092 |
| 2 | `<b>` | 2,529,454 | 12.77% | 1,947,686 | 217,847 | 363,921 |
| 3 | `.` | 1,452,856 | 7.34% | 1,116,654 | 126,425 | 209,777 |
| 4 | `8` | 1,337,595 | 6.75% | 1,032,619 | 112,770 | 192,206 |
| 5 | `16` | 1,001,659 | 5.06% | 773,616 | 84,246 | 143,797 |
| 6 | `-` | 851,566 | 4.30% | 679,659 | 76,094 | 95,813 |
| 7 | `4` | 820,827 | 4.14% | 629,639 | 72,143 | 119,045 |
| 8 | `<s>` | 760,879 | 3.84% | 585,742 | 65,451 | 109,686 |
| 9 | `#` | 691,699 | 3.49% | 558,538 | 60,732 | 72,429 |
| 10 | `=` | 542,179 | 2.74% | 417,278 | 46,700 | 78,201 |
| 11 | `L` | 329,964 | 1.67% | 253,945 | 28,474 | 47,545 |
| 12 | `J` | 293,947 | 1.48% | 226,485 | 25,191 | 42,271 |
| 13 | `r` | 272,660 | 1.38% | 209,773 | 23,120 | 39,767 |
| 14 | `JJ` | 228,196 | 1.15% | 176,232 | 19,183 | 32,781 |
| 15 | `LL` | 226,925 | 1.15% | 175,391 | 19,010 | 32,524 |
| 16 | `e` | 166,090 | 0.84% | 126,402 | 14,377 | 25,311 |
| 17 | `c` | 165,808 | 0.84% | 127,924 | 14,070 | 23,814 |
| 18 | `g` | 165,684 | 0.84% | 127,627 | 14,072 | 23,985 |
| 19 | `d` | 165,015 | 0.83% | 126,509 | 14,164 | 24,342 |
| 20 | `f` | 164,532 | 0.83% | 127,837 | 14,202 | 22,493 |
| 21 | `b` | 163,421 | 0.83% | 125,530 | 14,163 | 23,728 |
| 22 | `B` | 162,808 | 0.82% | 124,544 | 14,322 | 23,942 |
| 23 | `a` | 162,682 | 0.82% | 126,173 | 13,674 | 22,835 |
| 24 | `cc` | 158,735 | 0.80% | 120,983 | 13,249 | 24,503 |
| 25 | `dd` | 153,028 | 0.77% | 116,937 | 12,952 | 23,139 |

## Slide-Ready Outputs

- `data_preprocessing_dashboard.png`: compact summary for one slide.
- `dataset_split_counts.png`: train/validation/test counts.
- `token_length_distribution.png`: target sequence length distribution.
- `preprocessed_image_shapes.png`: exact cached image shape statistics with sampled scatter.
- `top_bekrn_tokens.png`: most frequent tokens.
- `composer_distribution.png`: composer composition by split.
- `preprocessing_examples.png`: raw score image beside cached model input.
- `preprocessing_pipeline.png`: pipeline diagram.
- `split_statistics.csv`, `top_tokens.csv`, `composer_counts.csv`: tables for slides/report.

## Presentation Notes

- The model task is image-conditioned symbolic transcription: score image in, BEKRN/KERN token sequence out.
- All dataset statistics above use the exact local partition files, not the partial CameraPrIMuS folder.
- The CRNN/CNNT runs used clean GrandStaff images. Distorted image files exist locally, but the active configs disabled them.
- The cache count equals the partition total, so the image-shape statistics describe every model input sample.
