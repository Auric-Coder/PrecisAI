# HANDOFF: PrecisAI

**Environment:** `precisai` (Python 3.11.17, PyTorch with CUDA on NVIDIA RTX 4060).
- Execution command: `& "C:\Users\ACER\miniconda3\condabin\conda.bat" run -n precisai python ...`
- Pins in `requirements.txt` / `environment.yml`. Smoke checks use `--smoke`.

## Current State & Resume Point (as of 30 Oct 2026)

### Completed Work

1. **Quality Tuning (Repetition, `<unk>` Elimination, Length Optimization):**
   - **Greedy Tuned (`no_repeat_ngram=3` + Attention-based UNK Replacement):**
     - ROUGE-1: **28.99** (+3.81 pts over baseline)
     - ROUGE-2: **7.12** (+1.37 pts)
     - ROUGE-L: **19.38** (+1.26 pts)
     - UNK rate: **0.0000** (eliminated via top-attended source token replacement in `src/vocab.py`)
     - Trigram repeat rate: **0.038** (dropped from 91.6% in baseline greedy)
   - **Beam Search & Hyperparameter Sweep:**
     - Tuned beam-4 (`alpha=1.0`, `min_len=20`, `unk_rep=True`): R1=**30.61**, R2=**8.86**, RL=**20.40**, avg_len=45.7, unk=0.0000.
     - Tuned beam-6 (`alpha=1.0`, `min_len=20`, `unk_rep=True`): R1=30.55, R2=**9.17**, RL=**20.41**, avg_len=44.7, unk=0.0000.
     - Documented in `results/tuning.md` and `results/beam_sweep.json`.
     - `src/decode.py` and `src/evaluate.py` updated with `no_repeat_ngram`, `cov_penalty`, `unk_replace`, and beam parameters with backward compatibility.

2. **Government Circulars & Regulatory Benchmark (`data/circulars/`, `src/gov_*`):**
   - 9 official press releases and regulatory instruments collected from PIB and relevant ministries.
   - Provenance, document metadata, and extraction tracking recorded in `data/circulars/manifest.jsonl`.
   - Clean UTF-8 texts in `data/circulars/cleaned/` and samples in `data/circulars/samples/`.
   - `src/gov_sources.py`: Scrapes and extracts verified text from official sources.
   - `src/gov_eval.py`: Evaluates zero-shot transfer under domain shift with sentence-aware chunking (`--chunking`) and lexical policy preservation audits (`factual_audit`).
   - Results documented in `results/government/`.

3. **Dual-Mode Streamlit Web UI (`app.py`):**
   - CNN/DailyMail interactive mode with TextRank vs. Seq2Seq comparison and attention heatmaps.
   - Government document mode with metadata selector, live token compression stats, truncation alerts, and policy preservation heuristic audit flags.

4. **Documentation & Repository Hygiene:**
   - Comprehensive `README.md` updated with full system architecture, math formulations, benchmark tables, visualizations, and quickstart commands.
   - Standard `LICENSE` (MIT) added.
   - `.gitignore` updated with `scratch/` and standard Python ignores.

### Remaining Optional Items:
- **Item 2:** Formal attention ablation write-up (Bahdanau vs Luong vs None comparison).
- **Item 3:** Detailed error analysis (10 good, 10 bad test summaries categorized in `results/error_analysis.md`).
- **Item 7:** Generate `VIVA.md` with 25 code-grounded viva Q&As.

## Existing CNN/DailyMail & Government Setup
- 40k/2k/500 split in `data/cache/`. Checkpoints in `checkpoints/` (`bahdanau.pt`, `luong.pt`, `none.pt`).
- Baseline metrics in `results/metrics.md`.
- Tuning metrics in `results/tuning.md`.
- Government benchmark artifacts in `results/government/`.

## UI/docs polish (31 Oct 2026)
- README fixed: single Beam 4 config in highlights, "zero-shot transfer under domain shift" wording, Limitations section, repo URL `Auric-Coder`. No Demo section (no results/ui_*.png).
- `app.py` rewritten as tabs (Summarize/Compare/Attention/About) with sidebar decoding controls; model calls unchanged. `plot_attention` gained optional `fontsize`/`colorbar` args (defaults unchanged).
- Verified with AppTest (load, TextRank, Bahdanau beam/greedy summarize, Compare): pass.
- README note: tuning.md attributes Luong's lag to unscaled dot products; README Limitations says unexplained/uninvestigated. Reconcile if desired.

## Syllabus extras (1 Nov 2026)
Core frozen (no edits to src/*.py, checkpoints, results/*). All new work is in `src/extras/` -> `results/extras/`. Extra pip deps installed in `precisai`: gensim, transformers, sentencepiece, sacrebleu (+ nltk wordnet/stopwords data). Not yet added to requirements.txt/environment.yml.
- T1 `decoding_study` -> decoding.md/json (sampling, perplexity, BLEU-4, unsupported-number rate; greedy/beam rows reproduce metrics.md exactly).
- T2 `prep_study` -> prep_study.md. T3 `tokenizer_study` -> tokenizers.md.
- T4 `w2v_study` -> word2vec.md + plots (GloVe download worked).
- T5 `hf_baselines {zero,bert,ft,table}` -> hf_baselines.md; `transformer_demo` -> bert_attention.png. Fine-tuned T5 in checkpoints/extras/ (gitignored).
- T6 README sections added.
- T7 `neural_extractive` -> neural_extractive.md (3 seeds; beats TextRank, below Lead-3).
- T8 `human_eval make|aggregate`: results/extras/human_eval.csv has BLANK ratings for the user to fill; system key in human_eval_key.json (keep from raters). Run `aggregate` after filling.
- Run python via `C:\Users\ACER\miniconda3\envs\precisai\python.exe` when running jobs in parallel (`conda run` breaks/buffers); set PYTHONIOENCODING=utf-8.
- Caveat: t5-small pretraining mixture includes CNN/DM, so its "pretrained" row is not clean zero-shot.
