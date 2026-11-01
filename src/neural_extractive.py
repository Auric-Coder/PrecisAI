"""T7 (Unit III): SummaRuNNer-lite neural extractive summariser.
Sentence encoder (BiGRU vs BiLSTM over words) -> document-level BiGRU over sentences -> per-sentence logistic scorer.
Labels: top-3 sentences by ROUGE-1+ROUGE-2 F (own n-gram implementation, no stemming) vs the reference highlights = 1.
Trained on the same 40k training articles; evaluated on the 500 test articles. Writes results/neural_extractive.md."""
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from .evaluate import rouge
from .metrics_extra import RES as OUT, ROOT, load_test
from .textrank import split_sentences
from .vocab import PAD_ID, Vocab, tokenize

MAX_S, MAX_W, K, EPOCHS, SEEDS = 40, 50, 3, 3, [0, 1, 2]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ngram_f(a, b, n):
    ca, cb = Counter(zip(*[a[i:] for i in range(n)])), Counter(zip(*[b[i:] for i in range(n)]))
    ov = sum((ca & cb).values())
    return 0.0 if ov == 0 else 2 * ov / (sum(ca.values()) + sum(cb.values()))


def prepare(articles, refs, vocab):
    """-> ids int16 [N,MAX_S,MAX_W], mask [N,MAX_S], labels [N,MAX_S], sentence text lists."""
    N = len(articles)
    ids = torch.zeros(N, MAX_S, MAX_W, dtype=torch.int16)
    mask, lab, texts = torch.zeros(N, MAX_S, dtype=torch.bool), torch.zeros(N, MAX_S), []
    for i, (a, r) in enumerate(zip(articles, refs)):
        sents = [s.strip() for s in split_sentences(a) if s.strip()][:MAX_S]
        texts.append(sents)
        rt = tokenize(r)
        sc = []
        for j, s in enumerate(sents):
            t = tokenize(s)[:MAX_W]
            if t:
                ids[i, j, : len(t)] = torch.tensor(vocab.encode(t), dtype=torch.int16)
            mask[i, j] = True
            sc.append(ngram_f(t, rt, 1) + ngram_f(t, rt, 2))
        lab[i, np.argsort(sc)[::-1][:K].copy()] = 1.0
    return ids, mask, lab, texts


class Extractor(nn.Module):
    def __init__(self, V, cell, emb=128, h=128):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.emb = nn.Embedding(V, emb, padding_idx=PAD_ID)
        self.word = rnn(emb, h, batch_first=True, bidirectional=True)
        self.doc = nn.GRU(2 * h, h, batch_first=True, bidirectional=True)
        self.drop, self.out = nn.Dropout(0.3), nn.Linear(2 * h, 1)

    def forward(self, ids, mask):
        B, S, W = ids.shape
        x = ids.view(B * S, W).long()
        o, _ = self.word(self.drop(self.emb(x)))
        w = (x != PAD_ID).unsqueeze(-1)
        sent = o.masked_fill(~w, -1e4).max(1).values.masked_fill(~(w.any(1)), 0).view(B, S, -1)  # max-pool over words
        d, _ = self.doc(self.drop(sent))
        return self.out(self.drop(d)).squeeze(-1).masked_fill(~mask, -1e4)


def train_eval(cell, seed, tr, te, V):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Extractor(V, cell).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    ids, mask, lab = tr[0], tr[1], tr[2]
    for ep in range(EPOCHS):
        m.train()
        for b in torch.randperm(len(ids)).split(32):
            i, k, y = ids[b].to(DEV), mask[b].to(DEV), lab[b].to(DEV)
            loss = (F.binary_cross_entropy_with_logits(m(i, k), y, reduction="none") * k).sum() / k.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()
    with torch.no_grad():
        sc = torch.cat([m(te[0][b:b + 100].to(DEV), te[1][b:b + 100].to(DEV)).cpu() for b in range(0, len(te[0]), 100)])
    return sc


def main():
    vocab = Vocab.load(ROOT / "data" / "cache" / "vocab.json")
    train = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train").shuffle(seed=42).select(range(40000))
    tr = prepare(train["article"], train["highlights"], vocab)[:3]
    d = load_test()
    te_ids, te_mask, te_lab, te_texts = prepare(d["article"], d["ref"], vocab)
    refs, arts = d["ref"], d["article"]
    print("data ready", flush=True)

    def select(scores):
        return [" ".join(t[j] for j in sorted(np.argsort(-s[:len(t)].numpy())[:K])) for s, t in zip(scores, te_texts)]

    def f1(scores):
        tp = fp = fn = 0
        for s, t, y in zip(scores, te_texts, te_lab):
            pred = set(np.argsort(-s[:len(t)].numpy())[:K].tolist())
            gold = set(torch.nonzero(y).flatten().tolist())
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
        return 100 * 2 * tp / max(2 * tp + fp + fn, 1)

    rows = {"Lead-3": (None, select(torch.cat([torch.linspace(1, 0, MAX_S).repeat(len(te_texts), 1)]))),
            "Oracle (top-3 by label)": (None, select(te_lab * 1.0 + 0.0))}
    rows["Lead-3"] = (f1(torch.linspace(1, 0, MAX_S).repeat(len(te_texts), 1)), rows["Lead-3"][1])
    rows["Oracle (top-3 by label)"] = (100.0, rows["Oracle (top-3 by label)"][1])
    res = {k: [(v[0], rouge(v[1], refs))] for k, v in rows.items()}
    for cell in ["gru", "lstm"]:
        res[f"SummaRuNNer-lite, {cell.upper()} sentence encoder"] = []
        for seed in SEEDS:
            sc = train_eval(cell, seed, tr, (te_ids, te_mask), len(vocab))
            res[f"SummaRuNNer-lite, {cell.upper()} sentence encoder"].append((f1(sc), rouge(select(sc), refs)))
            print(cell, seed, res[f"SummaRuNNer-lite, {cell.upper()} sentence encoder"][-1], flush=True)
    json.dump(res, open(OUT / "neural_extractive.json", "w"), indent=1)
    ms = lambda xs: f"{np.mean(xs):.2f} ± {np.std(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"
    L = ["# Neural extractive summariser (Unit III)", "",
         f"SummaRuNNer-lite: word embedding (128) -> BiGRU/BiLSTM sentence encoder (max-pool) -> document BiGRU -> logistic sentence scorer. "
         f"Up to {MAX_S} sentences x {MAX_W} words per article; {EPOCHS} epochs, Adam 1e-3, batch 32, 40k training articles, CNN/DM test n=500. "
         f"Mean ± std over seeds {SEEDS}.", "",
         "Labels: per article, top-3 sentences by ROUGE-1 F + ROUGE-2 F against the reference (own n-gram implementation, no stemming) = 1. "
         "F1 = sentence-selection F1 (x100) of the predicted top-3 against these labels. ROUGE = official scorer on the 3 selected sentences.", "",
         "| system | selection F1 | R-1 | R-2 | R-L |", "|---|---|---|---|---|"]
    for k, runs in res.items():
        L.append(f"| {k} | {ms([r[0] for r in runs])} | " + " | ".join(ms([r[1][c] for r in runs]) for c in ["rouge1", "rouge2", "rougeL"]) + " |")
    L += ["", "Reference: TF-IDF TextRank R-1 35.80 / R-2 14.10 / R-L 23.38 (results/metrics.md). Oracle uses the labels, so it is an upper bound for this labelling scheme only."]
    (OUT / "neural_extractive.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
