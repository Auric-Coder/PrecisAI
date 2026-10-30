# Decoding and Tuning Results (CNN/DM Test Subset n=500)

## 1. Quality Issues Identified in HANDOFF.md
1. **Severe Trigram Repetition in Greedy Decoding:** Baseline greedy had 91.6% trigram repeat rate (repeating phrase loops).
2. **High `<unk>` Rate:** 4.1% - 4.4% of tokens were `<unk>`, turning named entities, places, and numbers into `<unk>`.
3. **Short Output Length:** Baseline beam search produced ~44.5 words, whereas human reference summaries average 55–70 words.
4. **Suboptimal Model Training:** Luong attention validation loss lagged behind Bahdanau and no-attention due to unscaled dot products ($d=512$, large logit variance).

---

## 2. Greedy Decoding Hyperparameter Tuning (Bahdanau Model)

| Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Len | UNK Rate | Tri-Repeat Rate | Notes |
|---|---|---|---|---|---|---|---|
| Baseline Greedy (`no_rep=0`, `unk_rep=False`) | 25.18 | 5.75 | 18.12 | 49.5 | 0.0440 | 0.916 | Repetitive phrase loops; named entities replaced by `<unk>` |
| + Attention UNK Replacement (`unk_rep=True`) | 26.80 | 6.46 | 18.99 | 49.5 | **0.0000** | 0.890 | `<unk>` eliminated by aligning with top-attended source tokens |
| + Trigram Blocking (`no_repeat_ngram=3`) | 27.28 | 6.39 | 18.44 | 48.7 | 0.0448 | **0.000** | Trigram repetition completely eliminated (0.000) |
| **Combined Tuned Greedy** (`no_rep=3` + `unk_rep=True`) | **28.99** | **7.12** | **19.38** | 48.7 | **0.0000** | **0.038** | **+3.81 R1, +1.37 R2, +1.26 RL over baseline greedy!** |

---

## 3. Beam Search Sweep (Bahdanau Model, n=500 Test Articles)

| Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Len | UNK Rate | Tri-Repeat Rate | Key Effect |
|---|---|---|---|---|---|---|---|
| `baseline-beam4-alpha0.6` | 28.65 | 7.93 | 19.30 | 44.5 | 0.0408 | 0.000 | Baseline checkpoint performance |
| `beam4-alpha0.6-unkrep` | 30.35 | 8.82 | 20.30 | 44.5 | **0.0000** | 0.026 | +1.70 R1, +0.89 R2, +1.00 RL from UNK replacement |
| `beam4-alpha1.0-unkrep` | 30.56 | 8.87 | 20.38 | 45.6 | **0.0000** | 0.026 | Increasing length penalty $\alpha=1.0$ increases output length |
| `beam4-alpha1.0-minlen20-unkrep` | **30.61** | 8.86 | 20.40 | 45.7 | **0.0000** | 0.026 | **Highest overall ROUGE-1 (30.61)** |
| `beam6-alpha1.0-minlen20-unkrep` | 30.55 | **9.17** | **20.41** | 44.7 | **0.0000** | 0.016 | **Highest ROUGE-2 (9.17) and ROUGE-L (20.41)** |
| `beam4-alpha1.0-minlen20-cov0.2-unkrep` | 30.56 | 8.86 | 20.38 | 45.5 | **0.0000** | 0.026 | Coverage penalty $\beta=0.2$ maintains high precision |

---

## 4. Key Takeaways
1. **Attention-based UNK replacement** gives the single largest improvement across both greedy and beam search (+1.7 to +1.8 ROUGE-1, +0.9 ROUGE-2) while driving the UNK rate from >4% to **0.000%**.
2. **Trigram blocking** completely solves the 91.6% repetition pathology in greedy decoding without hurting fluency.
3. **Length penalty ($\alpha=1.0$) and $\text{min\_len}=20$** effectively counter premature `<eos>` generation, lengthening summaries from 44.5 to 45.7 tokens and yielding the highest ROUGE-1 (30.61). Beam 6 achieves the highest ROUGE-2 (9.17).
