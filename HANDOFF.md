# HANDOFF: PrecisAI

**Conda env:** `precisai` (py3.11, torch 2.14.1+cu126, RTX 4060). Existing envs (base/cv_env/launchpad) lacked CUDA torch/datasets/rouge/nltk, so a new one was made. Pins: requirements.txt / environment.yml. Use `conda run -n precisai ...`.

## Done
- Data: CNN/DM 3.0.0 subset 40k/2k/500 (HF `abisee/cnn_dailymail`, shuffle seed 42), 400/60 tokens, 30k train-only vocab, cached in data/cache/ (gitignored).
- Models: TextRank (TF-IDF PageRank, top-3), BiGRU+GRU seq2seq with `--attn bahdanau|luong|none`; greedy + beam4 (GNMT lp 0.6, no-repeat-trigram, min_len 10).
- Trained all 3 variants, 10 epochs each (AMP, bs64, Adam 1e-3, ~30/28/18 min). Eval, loss curves, attention heatmap in results/. Streamlit app (app.py) written; NOT launched or browser-tested.

## Commands
```
conda run -n precisai python -m src.preprocess            # (--smoke)
conda run -n precisai python -m src.train --attn bahdanau # bahdanau|luong|none; see run_all.sh
conda run -n precisai python -m src.evaluate              # -> results/metrics.{json,md}
conda run -n precisai streamlit run app.py
```

## ROUGE F1 x100 (test n=500; results/metrics.md)
| model | R-1 | R-2 | R-L | unk | tri-rep |
|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.4 | 0 | 0.66 |
| bahdanau greedy | 25.2 | 5.8 | 18.1 | .044 | 0.92 |
| bahdanau beam4 | 28.7 | 7.9 | 19.3 | .041 | 0 |
| luong greedy / beam4 | 14.0 / 16.5 | 1.6 / 2.4 | 11.8 / 12.1 | .12 / .09 | .96 / 0 |
| none greedy / beam4 | 17.8 / 19.4 | 2.6 / 3.4 | 13.6 / 13.8 | .08 / .06 | .95 / 0 |

## Known issues
- Abstractive models trail TextRank by a wide margin; no copy mechanism, so names/numbers become `<unk>` (4-12% of output tokens).
- Greedy decoding repeats trigrams in ~92-96% of summaries (beam has the block, so 0 by construction).
- Val loss was still falling at epoch 10 for every variant (no early stop triggered): undertrained.
- Luong ended worse than `none` (val CE 5.22 vs 4.83 vs 4.19 bahdanau). Not diagnosed: likely slow optimisation of the post-GRU "general" score, not a verified conclusion. Treat the attention ablation ordering for luong as unreliable until retrained/debugged.
- ROUGE uses full reference highlights; models see 400 tokens, so long-article tail is invisible. Single seed, no variance estimates.
- Lowercased whitespace-joined output (no detokenisation). Circulars (data/circulars/) are empty; domain shift from CNN/DM is untested.

## TODO (priority)
1. Launch app.py and verify all model/decoding combos, attn=none path, empty/short input.
2. Train longer (20-30 epochs, LR decay) and debug Luong (lower LR, scaled score, input feeding).
3. Add pointer-generator/copy + coverage to fix `<unk>` and repetition.
4. Add Lead-3 baseline and multiple seeds for the ablation table.
5. Evaluate on real circulars once added to data/circulars/ (no reference summaries yet).
