"""Word2Vec (CBOW / Skip-Gram) on the 40k training articles: neighbours, analogies, PCA/t-SNE, comparison with the
seq2seq encoder embedding (read-only). Saves models to checkpoints/ and writes results/word2vec.md.
Also provides sentence_vec_fn(kind) used by textrank(repr='w2v'|'bert'). Run: python -m src.embeddings"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .metrics_extra import RES as OUT, ROOT
from .vocab import tokenize

SEED = 0
W2V_PATH = ROOT / "checkpoints" / "w2v_{}.model"
_VEC = {}


def train_w2v():
    """CBOW + Skip-Gram on the 40k training articles (same subset/seed as preprocess.py)."""
    from datasets import load_dataset
    from gensim.models import Word2Vec
    train = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train").shuffle(seed=42).select(range(40000))
    sents = [tokenize(a) for a in train["article"]]
    for name, sg in [("cbow", 0), ("skipgram", 1)]:
        m = Word2Vec(sents, vector_size=128, window=5, min_count=5, sg=sg, negative=5, epochs=3, workers=1, seed=SEED)
        m.save(str(W2V_PATH).format(name))
        print(name, "trained", len(m.wv), flush=True)


def load_w2v(name):
    import os
    from gensim.models import Word2Vec
    path = str(W2V_PATH).format(name)
    if not os.path.exists(path):
        train_w2v()
    return Word2Vec.load(path).wv


def sentence_vec_fn(kind):
    """kind: w2v (Skip-Gram, averaged word vectors) | bert (mean-pooled bert-base-uncased)."""
    if kind == "bert":
        from .pretrained import bert_vec_fn
        return bert_vec_fn()
    if "w2v" not in _VEC:
        _VEC["w2v"] = load_w2v("skipgram")
    kv = _VEC["w2v"]

    def f(s):
        vs = [kv[w] for w in tokenize(s) if w in kv]
        return np.mean(vs, 0) if vs else np.zeros(kv.vector_size)
    return f


class Seq2SeqEmb:
    """Minimal most_similar over the Bahdanau encoder embedding matrix (read-only checkpoint load)."""

    def __init__(self):
        ck = torch.load(ROOT / "checkpoints" / "bahdanau.pt", map_location="cpu", weights_only=False)
        self.itos = ck["itos"]
        self.key_to_index = {w: i for i, w in enumerate(self.itos)}
        W = ck["state"]["eemb.weight"].float().numpy()
        self.W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)

    def __contains__(self, w):
        return w in self.key_to_index

    def __getitem__(self, w):
        return self.W[self.key_to_index[w]]

    def most_similar(self, positive=(), negative=(), topn=10):
        q = sum(self[w] for w in positive) - sum(self[w] for w in negative)
        s = self.W @ (q / np.linalg.norm(q))
        for w in list(positive) + list(negative):
            s[self.key_to_index[w]] = -9
        s[:4] = -9  # special tokens
        return [(self.itos[i], float(s[i])) for i in np.argsort(-s)[:topn]]


def analogy_acc(kv):
    res = []
    for a, b, c, d in ANALOGIES:
        if all(w in kv for w in (a, b, c, d)):
            top = [w for w, _ in kv.most_similar(positive=[b, c], negative=[a], topn=5)]
            res.append((f"{a}:{b}::{c}:{d}", top[0], d == top[0], d in top))
        else:
            res.append((f"{a}:{b}::{c}:{d}", "(OOV)", False, False))
    return res



def main():
    models = {n: type("M", (), {"wv": load_w2v(n)}) for n in ("cbow", "skipgram")}
    s2s = Seq2SeqEmb()
    embs = {"cbow": models["cbow"].wv, "skipgram": models["skipgram"].wv, "seq2seq-enc-emb": s2s}

    L = ["# Word2Vec study (Unit II)", "",
         "Gensim Word2Vec on the 40k CNN/DM training articles (project tokenizer, lowercased): dim 128, window 5, min_count 5, negative 5, 3 epochs, seed 0, workers=1. "
         f"Vocab {len(models['cbow'].wv)} (CBOW/SG). seq2seq-enc-emb = encoder embedding of checkpoints/bahdanau.pt (256-d, trained only for summarisation, read-only).", "",
         "## Nearest neighbours (cosine, top 6)", "", "| word | CBOW | Skip-Gram | seq2seq enc. emb |", "|---|---|---|---|"]
    overlap = {"cbow": [], "skipgram": []}
    for w in WORDS:
        row = []
        nn = {}
        for k, kv in embs.items():
            if w in kv:
                nn[k] = [x for x, _ in kv.most_similar(w, topn=10)] if k != "seq2seq-enc-emb" else [x for x, _ in kv.most_similar([w], topn=10)]
            row.append(", ".join(nn[k][:6]) if k in nn else "(OOV)")
        for k in overlap:
            if k in nn and "seq2seq-enc-emb" in nn:
                overlap[k].append(len(set(nn[k]) & set(nn["seq2seq-enc-emb"])) / 10)
        L.append(f"| {w} | " + " | ".join(row) + " |")
    L += ["", "Mean top-10 neighbour overlap with the seq2seq embedding: " + ", ".join(f"{k} = {np.mean(v):.2f}" for k, v in overlap.items()) + f" (over {len(overlap['cbow'])} words).", "",
          "## Analogies (a:b::c:?, top-1 / top-5 over the full vocab, honest small-corpus result)", "", "| analogy | " + " | ".join(embs) + " |", "|---|---|---|---|"]
    res = {k: analogy_acc(kv) for k, kv in embs.items()}
    for i, (a, b, c, d) in enumerate(ANALOGIES):
        L.append(f"| {a}:{b}::{c}:{d} | " + " | ".join(f"{res[k][i][1]} {'✓' if res[k][i][2] else ('~' if res[k][i][3] else '✗')}" for k in embs) + " |")
    L += ["", "✓ = correct top-1, ~ = correct in top-5, ✗ = miss. Accuracy top-1 / top-5: " + "; ".join(
        f"{k} {sum(r[2] for r in v)}/{len(v)} , {sum(r[3] for r in v)}/{len(v)}" for k, v in res.items()) + ".", ""]

    # PCA / t-SNE plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 15))
    for r, (k, kv) in enumerate(embs.items()):
        ws = [(w, g) for g, l in PLOT.items() for w in l if w in kv]
        X = np.stack([kv[w] for w, _ in ws])
        for c, (nm, proj) in enumerate([("PCA", PCA(2, random_state=SEED)), ("t-SNE", TSNE(2, perplexity=8, random_state=SEED, init="pca"))]):
            Y = proj.fit_transform(X)
            ax = axes[r, c]
            for gi, g in enumerate(PLOT):
                idx = [i for i, (_, gg) in enumerate(ws) if gg == g]
                ax.scatter(Y[idx, 0], Y[idx, 1], label=g, color=f"C{gi}")
                for i in idx:
                    ax.annotate(ws[i][0], Y[i], fontsize=7)
            ax.set_title(f"{k} - {nm}")
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "word2vec_plots.png", dpi=130)
    L += ["![PCA and t-SNE](word2vec_plots.png)", ""]

    L += ["W2V-TextRank (extractive) ROUGE is produced by `python -m src.evaluate --repr w2v` (see results/metrics.md)."]
    (OUT / "word2vec.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
