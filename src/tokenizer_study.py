"""T3: word-level (project) vs BERT WordPiece vs GPT-2 BPE vs T5 SentencePiece. Writes results/tokenizers.md."""
import json
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from .metrics_extra import RES as OUT, ROOT, gov_docs, load_test
from .vocab import Vocab, tokenize

LIMIT = 400
RARE = ["notification", "ordinance", "disbursement", "gazette", "amendment", "notwithstanding", "subsidiary", "ministry"]


def word_level_vocab():
    return Vocab.load(ROOT / "data" / "cache" / "vocab.json")


def main():
    arts = load_test()["article"]
    gov = list(gov_docs().values())
    vocab = word_level_vocab()
    toks = {
        "word-level (30k, project)": tokenize,
        "BERT WordPiece (bert-base-uncased)": AutoTokenizer.from_pretrained("bert-base-uncased").tokenize,
        "GPT-2 BPE": AutoTokenizer.from_pretrained("gpt2").tokenize,
        "T5 SentencePiece (t5-small)": AutoTokenizer.from_pretrained("t5-small").tokenize,
    }
    L = ["# Tokenizer study", "",
         "CNN/DM test (n=500 articles) and the 9 government documents. OOV rate: word-level = share of tokens not in the 30k vocab; "
         "subword tokenizers have no OOV by construction (BERT: `[UNK]` share reported; GPT-2 byte-level and T5 byte-fallback/`<unk>` share reported).", "",
         "| tokenizer | CNN/DM OOV/unk rate | CNN/DM avg tokens/article | CNN/DM % > 400 | gov OOV/unk rate | gov avg tokens/doc | gov % > 400 |", "|---|---|---|---|---|---|---|"]
    cols = {}
    for name, tk in toks.items():
        r = []
        for corpus in (arts, gov):
            seqs = [tk(a) for a in corpus]
            n = sum(map(len, seqs))
            if name.startswith("word"):
                unk = sum(t not in vocab.stoi for s in seqs for t in s)
            else:
                unk = sum(t in ("[UNK]", "<unk>") for s in seqs for t in s)
            r += [f"{100 * unk / n:.2f}%", f"{np.mean([len(s) for s in seqs]):.0f}", f"{100 * np.mean([len(s) > LIMIT for s in seqs]):.1f}%"]
        L.append(f"| {name} | " + " | ".join(r) + " |")
    # rare gov words: pick the least frequent long words in gov docs that the word vocab misses, plus fixed legal terms
    cnt = Counter(t for d in gov for t in tokenize(d) if t.isalpha())
    oov = [w for w, c in cnt.most_common() if w not in vocab.stoi and len(w) >= 8]
    words = (oov[:5] if len(oov) >= 5 else oov + [w for w in RARE if w in cnt and w not in oov])[:5]
    L += ["", "## Example segmentations of rare gov-doc words (OOV for the 30k word vocab where marked)", "",
          "| word | in 30k vocab | " + " | ".join(list(toks)[1:]) + " |", "|---|---|---|---|---|"]
    for w in words:
        L.append(f"| {w} | {'yes' if w in vocab.stoi else '**no** (-> `<unk>`)'} | " + " | ".join("`" + " ".join(tk(w)) + "`" for n, tk in list(toks.items())[1:]) + " |")
    L += ["", "Words chosen: the 5 most frequent alphabetic gov-doc words (>= 8 chars) missing from the word vocab"
          + (" (padded with fixed legal terms)." if len(oov) < 5 else "."),
          "", "Note: counts above are before truncation; the word-level seq2seq truncates to 400 *word* tokens, so a subword model would see fewer words per 400 tokens."]
    (OUT / "tokenizers.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
