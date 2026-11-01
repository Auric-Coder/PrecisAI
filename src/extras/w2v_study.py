"""T4: Word2Vec (CBOW / Skip-Gram) on the 40k training articles, neighbours, analogies, PCA/t-SNE,
comparison with the seq2seq encoder embedding (read-only), and W2V-TextRank. Writes results/extras/word2vec.md."""
import numpy as np
import torch
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..evaluate import rouge
from ..vocab import tokenize
from .common import ROOT, OUT, embedding_textrank, load_test

SEED = 0
WORDS = ["government", "president", "police", "court", "market", "election", "war", "health", "school", "football"]
ANALOGIES = [("man", "woman", "king", "queen"), ("he", "she", "his", "her"), ("france", "paris", "england", "london"),
             ("germany", "berlin", "italy", "rome"), ("good", "better", "bad", "worse"), ("monday", "tuesday", "friday", "saturday"),
             ("china", "chinese", "france", "french"), ("boy", "girl", "brother", "sister")]
PLOT = {"politics": ["government", "president", "minister", "election", "party", "vote", "law", "court"],
        "sport": ["football", "game", "team", "season", "club", "match", "league", "player"],
        "countries": ["britain", "france", "china", "russia", "india", "germany", "iran", "syria"],
        "time": ["monday", "tuesday", "friday", "sunday", "january", "june", "year", "week"]}


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
    train = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train").shuffle(seed=42).select(range(40000))
    sents = [tokenize(a) for a in train["article"]]
    models = {}
    for name, sg in [("cbow", 0), ("skipgram", 1)]:
        models[name] = Word2Vec(sents, vector_size=128, window=5, min_count=5, sg=sg, negative=5, epochs=3, workers=1, seed=SEED)
        print(name, "trained", len(models[name].wv), flush=True)
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

    # W2V-TextRank
    d = load_test()
    refs, arts = d["ref"], d["article"]

    def make_vec(kv):
        dim = kv.vector_size if hasattr(kv, "vector_size") else kv.W.shape[1]

        def f(s):
            vs = [kv[w] for w in tokenize(s) if w in kv]
            return np.mean(vs, 0) if vs else np.zeros(dim)
        return f
    L += ["## W2V-TextRank (extractive, top-3 sentences, averaged word vectors, cosine graph, PageRank d=0.85; CNN/DM test n=500)", "",
          "| system | R-1 | R-2 | R-L |", "|---|---|---|---|"]
    systems = {"W2V-TextRank (Skip-Gram)": embs["skipgram"], "W2V-TextRank (CBOW)": embs["cbow"], "seq2seq-embedding TextRank": s2s}
    try:
        import gensim.downloader as api
        systems["GloVe-100 TextRank"] = api.load("glove-wiki-gigaword-100")
    except Exception as e:
        L_note = f"GloVe skipped ({type(e).__name__})."
        print(L_note)
    for name, kv in systems.items():
        r = rouge([" ".join(embedding_textrank(a, make_vec(kv))) for a in arts], refs)
        L.append(f"| {name} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} |")
        print(name, r, flush=True)
    L += ["", "Reference TF-IDF TextRank (results/metrics.md): R-1 35.8 / R-2 14.1 / R-L 23.38."]
    if "GloVe-100 TextRank" not in systems:
        L.append("GloVe skipped: download unavailable.")
    (OUT / "word2vec.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
