# PrecisAI

Extractive (TextRank) + abstractive (BiGRU encoder / GRU decoder, from scratch) summarizer for long news / circulars. CNN/DailyMail 3.0.0 subset: 40k train / 2k val / 500 test, article truncated to 400 tokens, summary to 60, 30k vocab (train only).

## Setup
```
conda create -n precisai python=3.11 -y
conda run -n precisai pip install torch --index-url https://download.pytorch.org/whl/cu126
conda run -n precisai pip install -r requirements.txt   # pinned versions
```

## Run
```
python -m src.preprocess                  # add --smoke for 500/100/50 pipeline check
python -m src.train --attn bahdanau       # bahdanau | luong | none  (AMP, early stopping)
python -m src.evaluate                    # -> results/metrics.json, metrics.md, loss_curves.png, attention_heatmap.png
streamlit run app.py
```
`--smoke` on preprocess/train/evaluate uses separate dirs (`data/cache_smoke`, `checkpoints_smoke`, `results_smoke`).

## Models
- `textrank.py`: TF-IDF cosine graph over sentences, PageRank, top-3 sentences in document order.
- `model.py`: BiGRU encoder; GRUCell decoder; attention `bahdanau` (additive, context fed into GRU), `luong` (general, post-GRU), `none` (plain enc-dec).
- `decode.py`: batched greedy; beam search (beam 4, GNMT length penalty alpha 0.6, no-repeat-trigram, min length 10).

## Results (CNN/DM test subset, 500 articles, single seed)
| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat |
|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 |
| seq2seq-bahdanau-greedy | 25.18 | 5.75 | 18.12 | 49.5 | 0.044 | 0.916 |
| seq2seq-bahdanau-beam4 | 28.65 | 7.93 | 19.3 | 44.5 | 0.0408 | 0.0 |
| seq2seq-luong-greedy | 14.02 | 1.57 | 11.83 | 39.3 | 0.1161 | 0.962 |
| seq2seq-luong-beam4 | 16.49 | 2.44 | 12.07 | 30.7 | 0.0925 | 0.0 |
| seq2seq-none-greedy | 17.76 | 2.59 | 13.62 | 42.5 | 0.0805 | 0.95 |
| seq2seq-none-beam4 | 19.37 | 3.41 | 13.81 | 34.3 | 0.0584 | 0.0 |

![loss](results/loss_curves.png)
![attention](results/attention_heatmap.png)

TextRank beats all from-scratch seq2seq variants; models are undertrained (val loss still falling at epoch 10), have no copy mechanism (`<unk>` in 4-12% of tokens), and Luong underperforms. See HANDOFF.md for known issues and TODOs.

## Eval notes
ROUGE F1 (rouge-score, stemmed) against the full reference highlights. Extra columns: avg length, `<unk>` rate, share of summaries with a repeated trigram.

Checkpoints (~140 MB each) and the data cache are not in git; regenerate with the commands above.

Real circulars go in `data/circulars/` (left empty).
