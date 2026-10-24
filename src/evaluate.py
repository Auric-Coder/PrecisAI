import argparse
import json
import os
import time

import numpy as np
import torch
from rouge_score import rouge_scorer

from .dataset import dirs
from .decode import beam_search, greedy, load_model, plot_attention
from .textrank import textrank
from .vocab import UNK, tokenize

SRC_LEN = 400
SC = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(preds, refs):
    agg = np.zeros(3)
    for p, r in zip(preds, refs):
        s = SC.score(r, p)
        agg += [s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure]
    return dict(zip(["rouge1", "rouge2", "rougeL"], (100 * agg / len(preds)).round(2).tolist()))


def stats(preds):
    toks = [p.split() for p in preds]
    n = max(sum(map(len, toks)), 1)
    rep = np.mean([len(set(zip(t, t[1:], t[2:]))) < max(len(t) - 2, 0) for t in toks])
    return dict(avg_len=round(n / len(preds), 1), unk_rate=round(sum(t.count(UNK) for t in toks) / n, 4),
                trigram_repeat_rate=round(float(rep), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="+", default=["bahdanau", "luong", "none"])
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    cache, ckpt_dir, res = dirs(a.smoke)
    os.makedirs(res, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(os.path.join(cache, "test.pt"), weights_only=False)
    refs, arts, N = d["ref"], d["article"], len(d["ref"])
    metrics, preds_all = {}, {}

    def record(name, preds):
        metrics[name] = {**rouge(preds, refs), **stats(preds)}
        preds_all[name] = preds
        print(name, metrics[name], flush=True)

    record("textrank", [" ".join(textrank(t)) for t in arts])

    for m in a.models:
        path = os.path.join(ckpt_dir, f"{m}.pt")
        if not os.path.exists(path):
            print("skip (no checkpoint)", m)
            continue
        model, vocab = load_model(path, dev)
        t0 = time.time()
        g = []
        for i in range(0, N, a.bs):
            src, sl = d["src"][i:i + a.bs].to(dev), d["src_len"][i:i + a.bs].to(dev)
            src = src[:, : sl.max()]
            g += [vocab.decode(x) for x in greedy(model, src, sl)[0]]
        record(f"seq2seq-{m}-greedy", g)
        b = []
        for i in range(N):
            sl = d["src_len"][i:i + 1].to(dev)
            src = d["src"][i:i + 1, : sl.item()].to(dev)
            b.append(vocab.decode(beam_search(model, src, sl)[0]))
        record(f"seq2seq-{m}-beam4", b)
        print(f"  decode time {time.time()-t0:.0f}s")
        if m == "bahdanau":  # attention heatmap for the first test article (greedy)
            sl = d["src_len"][:1].to(dev)
            src = d["src"][:1, : sl.item()].to(dev)
            ids, att = greedy(model, src, sl)
            ids = ids[0]
            if att is not None and ids:
                plot_attention(att[0], tokenize(arts[0])[:SRC_LEN], [vocab.itos[i] for i in ids],
                               os.path.join(res, "attention_heatmap.png"))

    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(dict(n_test=N, metrics=metrics), f, indent=1)
    cols = ["rouge1", "rouge2", "rougeL", "avg_len", "unk_rate", "trigram_repeat_rate"]
    lines = [f"ROUGE F1 (x100), CNN/DM test subset n={N}", "", "| model | R-1 | R-2 | R-L | avg len | unk rate | tri-repeat |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | " + " | ".join(str(v[c]) for c in cols) + " |" for k, v in metrics.items()]
    with open(os.path.join(res, "metrics.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    samples = [dict(ref=refs[i], **{k: v[i] for k, v in preds_all.items()}) for i in range(min(3, N))]
    with open(os.path.join(res, "samples.json"), "w") as f:
        json.dump(samples, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for m in ["bahdanau", "luong", "none"]:
        p = os.path.join(res, f"loss_{m}.json")
        if os.path.exists(p):
            h = json.load(open(p))
            e = [x["epoch"] for x in h]
            l, = plt.plot(e, [x["train"] for x in h], label=f"{m} train")
            plt.plot(e, [x["val"] for x in h], "--", color=l.get_color(), label=f"{m} val")
    plt.xlabel("epoch"); plt.ylabel("token CE"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(res, "loss_curves.png"), dpi=140)


if __name__ == "__main__":
    main()
