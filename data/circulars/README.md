# Government document benchmark

This benchmark is independent of CNN/DailyMail and its train/validation/test caches. `raw/` contains the retrieved official HTML, `cleaned/` contains UTF-8 article text extracted from each PIB page, `references/` is reserved for reviewed labels, and `samples/` contains three source and reference-status examples. `manifest.jsonl` records provenance and extraction status.

## Collection status

- 9 PIB press-release documents were downloaded and text-extracted successfully.
- The Ministry of Education URL is a directory/discovery page, not a document; it is retained as a discovery record and excluded from evaluation.
- Every collected document is correctly typed as a **press release**. The PIB pages describe some guidelines and policies but are not themselves the complete instruments. Full official PDF links were identified for the Critical Mineral Recycling Guidelines, India AI Governance Guidelines, and affordable housing report; these PDFs have not yet been downloaded or extracted. The PM-Surya Ghar signed PDF link was expired (403).
- No human-reviewed reference summaries are present. `reference_summary` is null and status is pending. Generated summaries must not be treated as labels.
- Retrieval date: 2026-10-29. Respect each originating site's terms and rate limits when refreshing sources.

## Commands

```powershell
# Download the listed official pages and extract them; failures are logged per source.
python -m src.gov_sources --refresh

# Use existing cleaned documents and existing trained checkpoints.
python -m src.gov_eval

# Optional sentence-aware chunks for long documents; each chunk stays within its document's evaluation unit.
python -m src.gov_eval --chunking

# Choose checkpoint variants and decoder explicitly.
python -m src.gov_eval --models bahdanau luong none --decode both
```

Government evaluation writes only to `results/government/`. It uses the existing 400-token input and 60-token output limits, reports truncation, and computes no ROUGE when references are absent. `--chunking` processes every source sentence in separate chunks (oversized sentences are split at token boundaries); all chunks remain grouped under one `government_evaluation` split. Policy preservation checks are lexical heuristics, not factual verification. Never use these documents to train or select CNN/DailyMail checkpoints.
