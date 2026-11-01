# Neural extractive summariser (Unit III)

SummaRuNNer-lite: word embedding (128) -> BiGRU/BiLSTM sentence encoder (max-pool) -> document BiGRU -> logistic sentence scorer. Up to 40 sentences x 50 words per article; 3 epochs, Adam 1e-3, batch 32, 40k training articles, CNN/DM test n=500. Mean ± std over seeds [0, 1, 2].

Labels: per article, top-3 sentences by ROUGE-1 F + ROUGE-2 F against the reference (own n-gram implementation, no stemming) = 1. F1 = sentence-selection F1 (x100) of the predicted top-3 against these labels. ROUGE = official scorer on the 3 selected sentences.

| system | selection F1 | R-1 | R-2 | R-L |
|---|---|---|---|---|
| Lead-3 | 34.73 | 40.23 | 17.31 | 24.78 |
| Oracle (top-3 by label) | 100.00 | 49.39 | 27.23 | 34.49 |
| SummaRuNNer-lite, GRU sentence encoder | 41.42 ± 0.14 | 38.66 ± 0.12 | 16.60 ± 0.06 | 24.62 ± 0.15 |
| SummaRuNNer-lite, LSTM sentence encoder | 41.80 ± 1.09 | 38.82 ± 0.41 | 16.75 ± 0.38 | 24.81 ± 0.23 |

Reference: TF-IDF TextRank R-1 35.80 / R-2 14.10 / R-L 23.38 (results/metrics.md). Oracle uses the labels, so it is an upper bound for this labelling scheme only.
