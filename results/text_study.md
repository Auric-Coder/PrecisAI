# TextRank preprocessing ablation (CNN/DM test, n=500, top-3 sentences)

Full 2x2x3x2 grid over `textrank(weighting=, stop_words=, norm=, ngram=)`. Cosine-similarity graph, PageRank d=0.85. Analyzer: lowercase alphanumeric tokens, sklearn English stop list, NLTK Porter / WordNet lemmatizer.

| weighting | stop-words removed | normalisation | n-grams | R-1 | R-2 | R-L |
|---|---|---|---|---|---|---|
| bow | no | none | uni | 31.6 | 10.81 | 20.13 |
| bow | no | none | uni+bi | 31.6 | 11.04 | 20.45 |
| bow | no | porter | uni | 31.52 | 10.73 | 20.13 |
| bow | no | porter | uni+bi | 31.55 | 10.93 | 20.33 |
| bow | no | lemma | uni | 31.89 | 11.05 | 20.38 |
| bow | no | lemma | uni+bi | 31.42 | 10.93 | 20.3 |
| bow | yes | none | uni | 34.83 | 13.0 | 22.45 |
| bow | yes | none | uni+bi | 34.81 | 13.29 | 22.72 |
| bow | yes | porter | uni | 34.86 | 12.94 | 22.34 |
| bow | yes | porter | uni+bi | 34.75 | 13.25 | 22.62 |
| bow | yes | lemma | uni | 34.61 | 12.83 | 22.25 |
| bow | yes | lemma | uni+bi | 34.84 | 13.19 | 22.56 |
| tfidf | no | none | uni | 33.22 | 12.23 | 21.42 |
| tfidf | no | none | uni+bi | 33.01 | 12.51 | 21.73 |
| tfidf | no | porter | uni | 33.23 | 12.12 | 21.25 |
| tfidf | no | porter | uni+bi | 33.13 | 12.49 | 21.73 |
| tfidf | no | lemma | uni | 33.29 | 12.23 | 21.43 |
| tfidf | no | lemma | uni+bi | 33.26 | 12.75 | 21.94 |
| tfidf | yes | none | uni | 35.8 | 14.1 | 23.38 |
| tfidf | yes | none | uni+bi | 35.34 | 13.96 | 23.17 |
| tfidf | yes | porter | uni | 35.71 | 13.95 | 23.16 |
| tfidf | yes | porter | uni+bi | 35.26 | 14.0 | 23.2 |
| tfidf | yes | lemma | uni | 35.4 | 13.77 | 22.96 |
| tfidf | yes | lemma | uni+bi | 35.05 | 13.84 | 23.04 |

Marginal mean R-1 over the other factors:

- weighting: bow=33.19, tfidf=34.31
- stop-words removed: False=32.39, True=35.10
- normalisation: none=33.78, porter=33.75, lemma=33.72
- n-grams: (1, 1)=33.83, (1, 2)=33.67

Reference: default `textrank()` (TF-IDF, stop_words='english') in results/metrics.md.
