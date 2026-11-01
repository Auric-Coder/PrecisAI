# Tokenizer study

CNN/DM test (n=500 articles) and the 9 government documents. OOV rate: word-level = share of tokens not in the 30k vocab; subword tokenizers have no OOV by construction (BERT: `[UNK]` share reported; GPT-2 byte-level and T5 byte-fallback/`<unk>` share reported).

| tokenizer | CNN/DM OOV/unk rate | CNN/DM avg tokens/article | CNN/DM % > 400 | gov OOV/unk rate | gov avg tokens/doc | gov % > 400 |
|---|---|---|---|---|---|---|
| word-level (30k, project) | 2.44% | 812 | 84.8% | 9.53% | 2204 | 66.7% |
| BERT WordPiece (bert-base-uncased) | 0.00% | 867 | 88.6% | 0.00% | 2481 | 66.7% |
| GPT-2 BPE | 0.00% | 860 | 88.0% | 0.00% | 2559 | 66.7% |
| T5 SentencePiece (t5-small) | 0.00% | 971 | 91.2% | 0.00% | 2706 | 66.7% |

## Example segmentations of rare gov-doc words (OOV for the 30k word vocab where marked)

| word | in 30k vocab | BERT WordPiece (bert-base-uncased) | GPT-2 BPE | T5 SentencePiece (t5-small) |
|---|---|---|---|---|
| cooperatives | **no** (-> `<unk>`) | `cooperative ##s` | `co oper atives` | `▁cooperative s` |
| circularity | **no** (-> `<unk>`) | `circular ##ity` | `circ ularity` | `▁circular ity` |
| computerization | **no** (-> `<unk>`) | `computer ##ization` | `computer ization` | `▁computer ization` |
| molasses | **no** (-> `<unk>`) | `mo ##lass ##es` | `mol asses` | `▁ mol asse s` |
| chairmanship | **no** (-> `<unk>`) | `chairman ##ship` | `chair manship` | `▁chairman ship` |

Words chosen: the 5 most frequent alphabetic gov-doc words (>= 8 chars) missing from the word vocab.

Note: counts above are before truncation; the word-level seq2seq truncates to 400 *word* tokens, so a subword model would see fewer words per 400 tokens.
