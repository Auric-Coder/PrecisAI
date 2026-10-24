import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def split_sentences(text):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        import nltk
        try:
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def textrank(text, k=3, d=0.85, iters=100, tol=1e-6):
    """Return top-k sentences (in document order) ranked by PageRank over a TF-IDF cosine graph."""
    sents = [s.strip() for s in split_sentences(text) if s.strip()]
    if len(sents) <= k:
        return sents
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:
        return sents[:k]
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    rs = S.sum(1, keepdims=True)
    P = np.divide(S, rs, out=np.full_like(S, 1.0 / len(sents)), where=rs > 0)  # row-stochastic
    r = np.full(len(sents), 1.0 / len(sents))
    for _ in range(iters):
        nr = (1 - d) / len(sents) + d * P.T @ r
        if np.abs(nr - r).sum() < tol:
            r = nr
            break
        r = nr
    return [sents[i] for i in sorted(np.argsort(-r)[:k])]
