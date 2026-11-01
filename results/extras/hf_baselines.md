# Extractive / BiGRU / pretrained-transformer comparison

CNN/DM test n=500 (ROUGE F1 x100). Compression = mean source tokens (first 400 word tokens, mean 386) / mean summary tokens. Unsupported-number rate = share of numeric tokens in the summary absent from the source (T1 definition). Gov columns: 9 documents, no references, so no ROUGE; compression there is mean of per-document ratios (results/government definition).

| system | R-1 | R-2 | R-L | avg len | compression | unsup-num (CNN/DM) | unsup-num (gov) | gov compression |
|---|---|---|---|---|---|---|---|---|
| TextRank (TF-IDF) | 35.8 | 14.1 | 23.38 | 76.0 | 5.08 | 0.0 | 0/10 | 6.4061 |
| BiGRU Bahdanau greedy | 25.18 | 5.75 | 18.12 | 49.5 | 7.80 | 0.1895 | 1/4 | 7.0911 |
| BiGRU Bahdanau beam4 | 28.65 | 7.93 | 19.3 | 44.5 | 8.67 | 0.1484 | 0/9 | 8.3544 |
| BiGRU Bahdanau beam4 tuned | 30.61 | 8.86 | 20.4 | 45.7 | 8.45 | 0.1487 | - | - |
| BiGRU Luong beam4 | 16.49 | 2.44 | 12.07 | 30.7 | 12.57 | 0.7429 | 0/0 | 10.2626 |
| BiGRU no-attn beam4 | 19.37 | 3.41 | 13.81 | 34.3 | 11.25 | 0.7575 | 1/4 | 8.0094 |
| t5-small pretrained (zero-shot, beam4) | 37.29 | 16.54 | 26.51 | 45.1 | 8.56 | 0.0065 | 0/14 | 8.3589 |
| BERT-embedding TextRank | 32.84 | 11.24 | 20.13 | 106.8 | 3.61 | 0.0 | 0/38 | 2.5695 |
| t5-small fine-tuned (1 epoch, beam4) | 39.17 | 18.05 | 27.81 | 45.6 | 8.47 | 0.0016 | 0/17 | 7.7768 |

t5-small inputs are truncated to 400 SentencePiece tokens (~300 words), BiGRU inputs to 400 words; T5 uses `summarize: ` prefix, beam 4, max 60 new tokens. T5 outputs are lowercased/tokenised with the project tokenizer before scoring so the metrics are comparable. BERT-TextRank: bert-base-uncased last-layer mean-pooled sentence vectors (<=128 tokens), cosine graph, PageRank, top-3.
