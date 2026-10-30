Government evaluation: 9 documents, 9 evaluation units

ROUGE-1/2/L F1 not computed: no human-reviewed references are available.
All documents and their chunks belong to the single government_evaluation split.

| model | summaries | avg summary tokens | avg compression ratio | R-1 F1 | R-2 F1 | R-L F1 |
|---|---:|---:|---:|---:|---:|---:|
| textrank | 9 | 57.44 | 6.4061 | N/A | N/A | N/A |
| bigru_gru_bahdanau_greedy | 9 | 52.78 | 7.0911 | N/A | N/A | N/A |
| bigru_gru_bahdanau_beam | 9 | 46.44 | 8.3544 | N/A | N/A | N/A |
| bigru_gru_luong_greedy | 9 | 51.11 | 7.3058 | N/A | N/A | N/A |
| bigru_gru_luong_beam | 9 | 35.89 | 10.2626 | N/A | N/A | N/A |
| bigru_gru_none_greedy | 9 | 52.67 | 7.0119 | N/A | N/A | N/A |
| bigru_gru_none_beam | 9 | 46.67 | 8.0094 | N/A | N/A | N/A |

Input limit: 400 tokens per model call; summary limit: 60 tokens.
Over-limit documents are flagged in document_metrics.json. With --chunking, sentence-aware chunks are evaluated separately.
Policy preservation checks are lexical heuristics, not proof of factual correctness.
