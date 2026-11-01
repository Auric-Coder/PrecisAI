# TextRank preprocessing ablation (CNN/DM test, n=500, top-3 sentences)

Full 2x2x3x2 grid. Cosine-similarity graph, PageRank d=0.85 (as in `src/textrank.py`). Analyzer: lowercase alphanumeric tokens.

| weighting | stop-words removed | normalisation | n-grams | R-1 | R-2 | R-L |
|---|---|---|---|---|---|---|
| bow | no | none | uni | 31.6 | 10.81 | 20.13 |
| bow | no | none | uni+bi | 31.6 | 11.04 | 20.45 |
| bow | no | porter | uni | 31.52 | 10.73 | 20.13 |
| bow | no | porter | uni+bi | 31.55 | 10.93 | 20.33 |
| bow | no | lemma | uni | 31.89 | 11.05 | 20.38 |
| bow | no | lemma | uni+bi | 31.42 | 10.93 | 20.3 |
| bow | yes | none | uni | 35.52 | 13.65 | 22.92 |
| bow | yes | none | uni+bi | 35.28 | 13.85 | 23.09 |
| bow | yes | porter | uni | 35.14 | 13.39 | 22.77 |
| bow | yes | porter | uni+bi | 35.01 | 13.54 | 22.89 |
| bow | yes | lemma | uni | 35.13 | 13.35 | 22.88 |
| bow | yes | lemma | uni+bi | 35.08 | 13.7 | 23.02 |
| tfidf | no | none | uni | 33.22 | 12.23 | 21.41 |
| tfidf | no | none | uni+bi | 33.01 | 12.51 | 21.73 |
| tfidf | no | porter | uni | 33.23 | 12.12 | 21.26 |
| tfidf | no | porter | uni+bi | 33.13 | 12.49 | 21.73 |
| tfidf | no | lemma | uni | 33.29 | 12.23 | 21.43 |
| tfidf | no | lemma | uni+bi | 33.26 | 12.75 | 21.95 |
| tfidf | yes | none | uni | 35.84 | 14.24 | 23.5 |
| tfidf | yes | none | uni+bi | 35.44 | 14.22 | 23.4 |
| tfidf | yes | porter | uni | 35.75 | 14.15 | 23.31 |
| tfidf | yes | porter | uni+bi | 35.53 | 14.2 | 23.33 |
| tfidf | yes | lemma | uni | 35.49 | 14.09 | 23.34 |
| tfidf | yes | lemma | uni+bi | 35.36 | 14.22 | 23.35 |

Marginal mean R-1 over the other factors:

- weighting: bow=33.39, tfidf=34.38
- stop-words removed: False=32.39, True=35.38
- normalisation: none=33.94, porter=33.86, lemma=33.86
- n-grams: 1=33.97, 2=33.81

Reference: production `textrank.py` (sklearn TF-IDF, stop_words='english') scores R-1 35.8 in results/metrics.md.
