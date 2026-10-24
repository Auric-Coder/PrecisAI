ROUGE F1 (x100), CNN/DM test subset n=500

| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat |
|---|---|---|---|---|---|---|
| textrank | 35.8 | 14.1 | 23.38 | 76.0 | 0.0 | 0.664 |
| seq2seq-bahdanau-greedy | 25.18 | 5.75 | 18.12 | 49.5 | 0.044 | 0.916 |
| seq2seq-bahdanau-beam4 | 28.65 | 7.93 | 19.3 | 44.5 | 0.0408 | 0.0 |
| seq2seq-luong-greedy | 14.02 | 1.57 | 11.83 | 39.3 | 0.1161 | 0.962 |
| seq2seq-luong-beam4 | 16.49 | 2.44 | 12.07 | 30.7 | 0.0925 | 0.0 |
| seq2seq-none-greedy | 17.76 | 2.59 | 13.62 | 42.5 | 0.0805 | 0.95 |
| seq2seq-none-beam4 | 19.37 | 3.41 | 13.81 | 34.3 | 0.0584 | 0.0 |
