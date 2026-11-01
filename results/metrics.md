ROUGE F1 (x100), CNN/DM test subset n=500

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat |
|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 |
| seq2seq-bahdanau-greedy | 25.18 | 5.75 | 18.12 | 49.5 | 0.044 | 0.916 |
| seq2seq-bahdanau-beam4 | 28.65 | 7.93 | 19.3 | 44.5 | 0.0408 | 0.0 |
| seq2seq-luong-greedy | 14.02 | 1.57 | 11.83 | 39.3 | 0.1161 | 0.962 |
| seq2seq-luong-beam4 | 16.49 | 2.44 | 12.07 | 30.7 | 0.0925 | 0.0 |
| seq2seq-none-greedy | 17.76 | 2.59 | 13.62 | 42.5 | 0.0805 | 0.95 |
| seq2seq-none-beam4 | 19.37 | 3.41 | 13.81 | 34.3 | 0.0584 | 0.0 |

## Extended comparison (models=['bahdanau', 'luong', 'none', 't5-small', 'bert-extractive'] unk_replace=False beam=4 alpha=0.6 min_len=10 decode=None temperature=1.0 extra_metrics=True repr=['tfidf', 'w2v', 'bert'])

ROUGE F1 (x100), CNN/DM test subset n=500. Sampling rows: mean ± std over seeds [0, 1, 2] (top-k k=40, nucleus p=0.9). Unsupported-number rate = share of numeric tokens in the summary absent from the source (hallucination proxy).

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat | BLEU-4 | distinct-1 | distinct-2 | unsupported-number rate | test ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 | 10.29 | 0.263 | 0.717 | 0.0 | - |
| textrank-w2v | 31.27 | 10.06 | 18.96 | 96.1 | 0.0 | 0.34 | 6.56 | 0.253 | 0.733 | 0.0 | - |
| textrank-bert | 32.84 | 11.24 | 20.13 | 91.5 | 0.0 | 0.384 | 7.39 | 0.256 | 0.74 | 0.0 | - |
| seq2seq-bahdanau-greedy | 25.18 | 5.75 | 18.12 | 49.5 | 0.044 | 0.916 | 4.04 | 0.099 | 0.352 | 0.1895 | 72.03 |
| seq2seq-bahdanau-beam4 | 28.65 | 7.93 | 19.3 | 44.5 | 0.0408 | 0.0 | 5.14 | 0.121 | 0.453 | 0.1484 | 72.03 |
| seq2seq-luong-greedy | 14.02 | 1.57 | 11.83 | 39.3 | 0.1161 | 0.962 | 0.81 | 0.013 | 0.048 | 0.7956 | 188.78 |
| seq2seq-luong-beam4 | 16.49 | 2.44 | 12.07 | 30.7 | 0.0925 | 0.0 | 0.87 | 0.022 | 0.08 | 0.7429 | 188.78 |
| seq2seq-none-greedy | 17.76 | 2.59 | 13.62 | 42.5 | 0.0805 | 0.95 | 1.4 | 0.062 | 0.18 | 0.7667 | 129.49 |
| seq2seq-none-beam4 | 19.37 | 3.41 | 13.81 | 34.3 | 0.0584 | 0.0 | 1.44 | 0.083 | 0.245 | 0.7575 | 129.49 |
| t5-small-beam4 | 37.29 | 16.54 | 26.51 | 45.1 | 0.0 | 0.17 | 11.33 | 0.239 | 0.715 | 0.0065 | - |
| bert-extractive | 32.84 | 11.24 | 20.13 | 91.5 | 0.0 | 0.384 | 7.39 | 0.256 | 0.74 | 0.0 | - |

## Extended comparison (models=['bahdanau'] unk_replace=False beam=4 alpha=0.6 min_len=10 decode=topk temperature=1.0 extra_metrics=True repr=['tfidf'])

ROUGE F1 (x100), CNN/DM test subset n=500. Sampling rows: mean ± std over seeds [0, 1, 2] (top-k k=40, nucleus p=0.9). Unsupported-number rate = share of numeric tokens in the summary absent from the source (hallucination proxy).

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat | BLEU-4 | distinct-1 | distinct-2 | unsupported-number rate | test ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 | 10.29 | 0.263 | 0.717 | 0.0 | - |
| seq2seq-bahdanau-topk-T1.0 | 24.35 ± 0.24 | 3.61 ± 0.17 | 15.18 ± 0.21 | 53.23 ± 0.12 | 0.04 ± 0.00 | 0.26 ± 0.02 | 2.35 ± 0.09 | 0.13 ± 0.00 | 0.56 ± 0.00 | 0.45 ± 0.02 | 72.03 |

## Extended comparison (models=['bahdanau'] unk_replace=False beam=4 alpha=0.6 min_len=10 decode=nucleus temperature=1.0 extra_metrics=True repr=['tfidf'])

ROUGE F1 (x100), CNN/DM test subset n=500. Sampling rows: mean ± std over seeds [0, 1, 2] (top-k k=40, nucleus p=0.9). Unsupported-number rate = share of numeric tokens in the summary absent from the source (hallucination proxy).

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat | BLEU-4 | distinct-1 | distinct-2 | unsupported-number rate | test ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 | 10.29 | 0.263 | 0.717 | 0.0 | - |
| seq2seq-bahdanau-nucleus-T1.0 | 22.61 ± 0.05 | 3.10 ± 0.06 | 14.18 ± 0.05 | 53.83 ± 0.29 | 0.03 ± 0.00 | 0.18 ± 0.02 | 2.28 ± 0.13 | 0.17 ± 0.00 | 0.66 ± 0.00 | 0.53 ± 0.00 | 72.03 |

## Extended comparison (models=['bahdanau'] unk_replace=False beam=4 alpha=0.6 min_len=10 decode=topk temperature=0.7 extra_metrics=True repr=['tfidf'])

ROUGE F1 (x100), CNN/DM test subset n=500. Sampling rows: mean ± std over seeds [0, 1, 2] (top-k k=40, nucleus p=0.9). Unsupported-number rate = share of numeric tokens in the summary absent from the source (hallucination proxy).

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat | BLEU-4 | distinct-1 | distinct-2 | unsupported-number rate | test ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 | 10.29 | 0.263 | 0.717 | 0.0 | - |
| seq2seq-bahdanau-topk-T0.7 | 25.82 ± 0.22 | 4.91 ± 0.20 | 16.87 ± 0.19 | 51.07 ± 0.34 | 0.04 ± 0.00 | 0.50 ± 0.01 | 3.44 ± 0.20 | 0.12 ± 0.00 | 0.49 ± 0.00 | 0.36 ± 0.02 | 72.03 |

## Extended comparison (models=['bahdanau'] unk_replace=False beam=4 alpha=0.6 min_len=10 decode=nucleus temperature=0.7 extra_metrics=True repr=['tfidf'])

ROUGE F1 (x100), CNN/DM test subset n=500. Sampling rows: mean ± std over seeds [0, 1, 2] (top-k k=40, nucleus p=0.9). Unsupported-number rate = share of numeric tokens in the summary absent from the source (hallucination proxy).

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat | BLEU-4 | distinct-1 | distinct-2 | unsupported-number rate | test ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 | 10.29 | 0.263 | 0.717 | 0.0 | - |
| seq2seq-bahdanau-nucleus-T0.7 | 26.40 ± 0.08 | 5.28 ± 0.08 | 17.30 ± 0.07 | 51.03 ± 0.29 | 0.04 ± 0.00 | 0.55 ± 0.01 | 3.67 ± 0.02 | 0.12 ± 0.00 | 0.48 ± 0.00 | 0.32 ± 0.01 | 72.03 |
