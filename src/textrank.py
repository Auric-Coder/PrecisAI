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


_NORM = {}


def _normalizer(name):
    """none | porter | lemma (nltk; WordNet needs nltk.download('wordnet'))."""
    if name not in _NORM:
        if name == "porter":
            from nltk.stem import PorterStemmer
            _NORM[name] = PorterStemmer().stem
        elif name == "lemma":
            from nltk.stem import WordNetLemmatizer
            _NORM[name] = WordNetLemmatizer().lemmatize
        else:
            _NORM[name] = lambda w: w
    return _NORM[name]


def _lexical_vectors(sents, stop_words, norm, weighting, ngram):
    """Sentence-term matrix (BoW counts or TF-IDF) with optional stop-word removal, stemming/lemmatization, n-grams; rows L2-normalised."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from sklearn.preprocessing import normalize
    f = _normalizer(norm)

    def analyze(s):
        ws = [f(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if not (stop_words and w in ENGLISH_STOP_WORDS)]
        return [" ".join(ws[i:i + n]) for n in range(ngram[0], ngram[1] + 1) for i in range(len(ws) - n + 1)]
    vec = TfidfVectorizer(analyzer=analyze) if weighting == "tfidf" else CountVectorizer(analyzer=analyze)
    return normalize(vec.fit_transform(sents).astype(float))


def textrank(text, k=3, d=0.85, iters=100, tol=1e-6, repr="tfidf", stop_words=True, norm="none",
             weighting="tfidf", ngram=(1, 1), vec_fn=None):
    """Return top-k sentences (in document order) ranked by PageRank over a sentence-similarity graph.
    Defaults reproduce the original TF-IDF cosine graph. repr: tfidf | w2v | bert (averaged Word2Vec / mean-pooled BERT
    sentence vectors; vec_fn overrides the vectoriser). stop_words/norm/weighting/ngram only affect repr='tfidf'."""
    sents = [s.strip() for s in split_sentences(text) if s.strip()]
    if len(sents) <= k:
        return sents
    try:
        if repr == "tfidf" and stop_words and norm == "none" and weighting == "tfidf" and tuple(ngram) == (1, 1):
            X = TfidfVectorizer(stop_words="english").fit_transform(sents)
        elif repr == "tfidf":
            X = _lexical_vectors(sents, stop_words, norm, weighting, ngram)
        else:
            if vec_fn is None:
                from .embeddings import sentence_vec_fn
                vec_fn = sentence_vec_fn(repr)
            V = np.stack([vec_fn(s) for s in sents])
            n = np.linalg.norm(V, axis=1, keepdims=True)
            X = V / np.where(n > 0, n, 1)
    except ValueError:
        return sents[:k]
    S = (X @ X.T)
    S = S.toarray() if hasattr(S, "toarray") else np.clip(S, 0, None)
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
