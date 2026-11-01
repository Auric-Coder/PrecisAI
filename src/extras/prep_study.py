"""T2: TextRank preprocessing ablation on the 500 test articles.
Reuses textrank.split_sentences (textrank.py untouched); the PageRank below is the same as textrank.textrank,
only the sentence vectoriser is swapped."""
import itertools
import re

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..evaluate import rouge
from ..textrank import split_sentences
from .common import OUT, load_test

PS, LM, SW = PorterStemmer(), WordNetLemmatizer(), set(stopwords.words("english"))
TOK = re.compile(r"[a-z0-9]+")
NORM = {"none": lambda w: w, "porter": PS.stem, "lemma": LM.lemmatize}


def make_analyzer(norm, ngram, stop):
    f = NORM[norm]

    def analyze(s):
        ws = [f(w) for w in TOK.findall(s.lower()) if not (stop and w in SW)]
        return ws + ([a + "_" + b for a, b in zip(ws, ws[1:])] if ngram == 2 else [])
    return analyze


def rank(text, vec, k=3, d=0.85, iters=100, tol=1e-6):
    sents = [s.strip() for s in split_sentences(text) if s.strip()]
    if len(sents) <= k:
        return sents
    try:
        X = vec.fit_transform(sents).astype(float)
    except ValueError:
        return sents[:k]
    n = np.sqrt(X.multiply(X).sum(1)).A1  # L2-normalise (TfidfVectorizer already does; CountVectorizer does not)
    X = X.multiply(1 / np.where(n > 0, n, 1)[:, None]).tocsr()
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    rs = S.sum(1, keepdims=True)
    P = np.divide(S, rs, out=np.full_like(S, 1.0 / len(sents)), where=rs > 0)
    r = np.full(len(sents), 1.0 / len(sents))
    for _ in range(iters):
        nr = (1 - d) / len(sents) + d * P.T @ r
        done = np.abs(nr - r).sum() < tol
        r = nr
        if done:
            break
    return [sents[i] for i in sorted(np.argsort(-r)[:k])]


def main():
    d = load_test()
    refs, arts = d["ref"], d["article"]
    rows = []
    for weight, stop, norm, ng in itertools.product(["bow", "tfidf"], [False, True], ["none", "porter", "lemma"], [1, 2]):
        an = make_analyzer(norm, ng, stop)
        cls = CountVectorizer if weight == "bow" else TfidfVectorizer
        r = rouge([" ".join(rank(t, cls(analyzer=an))) for t in arts], refs)
        rows.append((weight, stop, norm, ng, r))
        print(weight, stop, norm, ng, r, flush=True)
    L = ["# TextRank preprocessing ablation (CNN/DM test, n=500, top-3 sentences)", "",
         "Full 2x2x3x2 grid. Cosine-similarity graph, PageRank d=0.85 (as in `src/textrank.py`). Analyzer: lowercase alphanumeric tokens.", "",
         "| weighting | stop-words removed | normalisation | n-grams | R-1 | R-2 | R-L |", "|---|---|---|---|---|---|---|"]
    L += [f"| {w} | {'yes' if s else 'no'} | {n} | {'uni' if g == 1 else 'uni+bi'} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} |"
          for w, s, n, g, r in rows]
    L += ["", "Marginal mean R-1 over the other factors:", ""]
    for name, i, vals in [("weighting", 0, ["bow", "tfidf"]), ("stop-words removed", 1, [False, True]),
                          ("normalisation", 2, ["none", "porter", "lemma"]), ("n-grams", 3, [1, 2])]:
        L.append(f"- {name}: " + ", ".join(f"{v}={np.mean([x[4]['rouge1'] for x in rows if x[i] == v]):.2f}" for v in vals))
    L += ["", "Reference: production `textrank.py` (sklearn TF-IDF, stop_words='english') scores R-1 35.8 in results/metrics.md."]
    (OUT / "prep_study.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
