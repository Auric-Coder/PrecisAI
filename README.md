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

## Eval notes
ROUGE F1 (rouge-score, stemmed) against the full reference highlights. Extra columns: avg length, `<unk>` rate, share of summaries with a repeated trigram.

Real circulars go in `data/circulars/` (left empty).
