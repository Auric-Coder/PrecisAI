"""Unit I: TextRank preprocessing ablation via textrank() options. Writes results/text_study.md.
Run: python -m src.text_study"""
import itertools

import numpy as np

from .evaluate import rouge
from .metrics_extra import RES as OUT, load_test
from .textrank import textrank


def main():
    d = load_test()
    refs, arts = d["ref"], d["article"]
    rows = []
    for weight, stop, norm, ng in itertools.product(["bow", "tfidf"], [False, True], ["none", "porter", "lemma"], [(1, 1), (1, 2)]):
        r = rouge([" ".join(textrank(t, weighting=weight, stop_words=stop, norm=norm, ngram=ng)) for t in arts], refs)
        rows.append((weight, stop, norm, ng, r))
        print(weight, stop, norm, ng, r, flush=True)
    L = ["# TextRank preprocessing ablation (CNN/DM test, n=500, top-3 sentences)", "",
         "Full 2x2x3x2 grid over `textrank(weighting=, stop_words=, norm=, ngram=)`. Cosine-similarity graph, PageRank d=0.85. "
         "Analyzer: lowercase alphanumeric tokens, sklearn English stop list, NLTK Porter / WordNet lemmatizer.", "",
         "| weighting | stop-words removed | normalisation | n-grams | R-1 | R-2 | R-L |", "|---|---|---|---|---|---|---|"]
    L += [f"| {w} | {'yes' if s else 'no'} | {n} | {'uni' if g == (1, 1) else 'uni+bi'} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} |"
          for w, s, n, g, r in rows]
    L += ["", "Marginal mean R-1 over the other factors:", ""]
    for name, i, vals in [("weighting", 0, ["bow", "tfidf"]), ("stop-words removed", 1, [False, True]),
                          ("normalisation", 2, ["none", "porter", "lemma"]), ("n-grams", 3, [(1, 1), (1, 2)])]:
        L.append(f"- {name}: " + ", ".join(f"{v}={np.mean([x[4]['rouge1'] for x in rows if x[i] == v]):.2f}" for v in vals))
    L += ["", "Reference: default `textrank()` (TF-IDF, stop_words='english') in results/metrics.md."]
    (OUT / "text_study.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
