import argparse
import json
import os
import time

import numpy as np
import torch
from rouge_score import rouge_scorer

from .dataset import dirs
from .decode import beam_search, greedy, load_model, plot_attention
from .metrics_extra import RES, extra_columns, perplexity
from .pretrained import HF_MODELS, summarize
from .sampling import sample
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
    ap.add_argument("--unk_replace", action="store_true", help="Replace <unk> tokens using top attention weights")
    ap.add_argument("--beam", type=int, default=4, help="Beam size")
    ap.add_argument("--alpha", type=float, default=0.6, help="GNMT length penalty")
    ap.add_argument("--min_len", type=int, default=10, help="Minimum sequence length")
    ap.add_argument("--no_repeat_ngram", type=int, default=3, help="N-gram blocking size for greedy")
    ap.add_argument("--cov_penalty", type=float, default=0.0, help="Coverage penalty coefficient")
    ap.add_argument("--decode", choices=["greedy", "beam", "topk", "nucleus"], default=None,
                    help="Run only this decoder (default: greedy and beam). topk/nucleus are sampled, mean +- std over --seeds")
    ap.add_argument("--topk", type=int, default=40, help="k for --decode topk")
    ap.add_argument("--top_p", type=float, default=0.9, help="p for --decode nucleus")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds for sampling decoders")
    ap.add_argument("--extra_metrics", action="store_true", help="Add test perplexity, BLEU-4, distinct-1/2 and unsupported-number rate columns")
    ap.add_argument("--repr", nargs="+", choices=["tfidf", "w2v", "bert"], default=["tfidf"], help="TextRank sentence representations")
    a = ap.parse_args()
    extended = bool(a.decode or a.extra_metrics or a.repr != ["tfidf"] or any(m in HF_MODELS for m in a.models))
    cache, ckpt_dir, res = dirs(a.smoke)
    os.makedirs(res, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(os.path.join(cache, "test.pt"), weights_only=False)
    refs, arts, N = d["ref"], d["article"], len(d["ref"])
    metrics, preds_all = {}, {}

    def save_partial(name):
        """Extended runs persist each row immediately so a late failure keeps earlier results."""
        if extended:
            os.makedirs(os.path.join(res, "partial"), exist_ok=True)
            with open(os.path.join(res, "partial", f"{name}.json"), "w") as f:
                json.dump(metrics[name], f)

    def record(name, preds, ppl=None):
        metrics[name] = {**rouge(preds, refs), **stats(preds)}
        if a.extra_metrics:
            metrics[name].update(extra_columns(preds, refs, arts))
            metrics[name]["ppl"] = ppl if ppl is not None else "-"
        save_partial(name)
        preds_all[name] = preds
        print(name, metrics[name], flush=True)

    for r in a.repr:
        record("textrank" if r == "tfidf" else f"textrank-{r}", [" ".join(textrank(t, repr=r)) for t in arts])

    for m in a.models:
        if m in HF_MODELS:
            raw = summarize(m, list(arts))
            record(m if m == "bert-extractive" else f"{m}-beam4", [" ".join(tokenize(x)) for x in raw] if m == "t5-small" else raw)
            continue
        path = os.path.join(ckpt_dir, f"{m}.pt")
        if not os.path.exists(path):
            print("skip (no checkpoint)", m)
            continue
        model, vocab = load_model(path, dev)
        ppl = perplexity(model, d, dev) if a.extra_metrics else None
        t0 = time.time()
        g = []
        if a.decode in ("topk", "nucleus"):
            sampled = []
            for seed in a.seeds:
                gen = torch.Generator(device=dev).manual_seed(seed)
                preds = []
                for i in range(0, N, a.bs):
                    src, sl = d["src"][i:i + a.bs].to(dev), d["src_len"][i:i + a.bs].to(dev)
                    out = sample(model, src[:, : sl.max()], sl, a.decode, a.temperature, a.topk, a.top_p, min_len=a.min_len, gen=gen)
                    preds += [vocab.decode(x) for x in out]
                sampled.append({**rouge(preds, refs), **stats(preds), **(extra_columns(preds, refs, arts) if a.extra_metrics else {})})
            name = f"seq2seq-{m}-{a.decode}-T{a.temperature}"
            metrics[name] = {c: f"{np.mean([r[c] for r in sampled]):.2f} ± {np.std([r[c] for r in sampled]):.2f}" for c in sampled[0]}
            if a.extra_metrics:
                metrics[name]["ppl"] = ppl
            save_partial(name)
            print(name, metrics[name], flush=True)
            continue
        if a.decode in (None, "greedy"):
            for i in range(0, N, a.bs):
                src, sl = d["src"][i:i + a.bs].to(dev), d["src_len"][i:i + a.bs].to(dev)
                src = src[:, : sl.max()]
                out_ids, atts = greedy(model, src, sl, min_len=a.min_len, no_repeat_ngram=a.no_repeat_ngram)
                for j in range(len(out_ids)):
                    raw_toks = tokenize(arts[i + j])[:SRC_LEN] if a.unk_replace else None
                    att_j = atts[j] if (a.unk_replace and atts is not None) else None
                    g.append(vocab.decode(out_ids[j], raw_toks, att_j))
            record(f"seq2seq-{m}-greedy", g, ppl)
        if a.decode in (None, "beam"):
            b = []
            for i in range(N):
                sl = d["src_len"][i:i + 1].to(dev)
                src = d["src"][i:i + 1, : sl.item()].to(dev)
                ids, att = beam_search(model, src, sl, beam=a.beam, alpha=a.alpha, min_len=a.min_len, cov_penalty=a.cov_penalty)
                raw_toks = tokenize(arts[i])[:SRC_LEN] if a.unk_replace else None
                b.append(vocab.decode(ids, raw_toks, att if a.unk_replace else None))
            record(f"seq2seq-{m}-beam{a.beam}", b, ppl)
        print(f"  decode time {time.time()-t0:.0f}s")
        if m == "bahdanau" and not extended:  # attention heatmap for the first test article (greedy)
            sl = d["src_len"][:1].to(dev)
            src = d["src"][:1, : sl.item()].to(dev)
            ids, att = greedy(model, src, sl)
            ids = ids[0]
            if att is not None and ids:
                plot_attention(att[0], tokenize(arts[0])[:SRC_LEN], [vocab.itos[i] for i in ids],
                               os.path.join(res, "attention_heatmap.png"))

    if extended:  # append-only: never overwrite the baseline tables/plots
        cols = ["rouge1", "rouge2", "rougeL", "avg_len", "unk_rate", "trigram_repeat_rate"]
        names = ["R-1", "R-2", "R-L", "avg len", "unk rate", "tri-repeat"]
        if a.extra_metrics:
            cols += ["bleu4", "distinct1", "distinct2", "unsup_num", "ppl"]
            names += ["BLEU-4", "distinct-1", "distinct-2", "unsupported-number rate", "test ppl"]
        cmd = " ".join(f"{k}={v}" for k, v in vars(a).items() if k in ("models", "decode", "repr", "extra_metrics", "temperature", "unk_replace", "beam", "alpha", "min_len"))
        lines = ["", f"## Extended comparison ({cmd})", "",
                 f"ROUGE F1 (x100), CNN/DM test subset n={N}. Sampling rows: mean ± std over seeds {a.seeds} (top-k k={a.topk}, nucleus p={a.top_p}). "
                 "Unsupported-number rate = share of numeric tokens in the summary absent from the source (hallucination proxy).", "",
                 "| model | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        lines += [f"| {k} | " + " | ".join(str(v.get(c, "-")) for c in cols) + " |" for k, v in metrics.items()]
        with open(os.path.join(res, "metrics.md"), "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        ej = os.path.join(res, "extended_metrics.json")
        old = json.load(open(ej)) if os.path.exists(ej) else {}
        old[cmd] = dict(n_test=N, metrics=metrics)
        json.dump(old, open(ej, "w"), indent=1)
        return

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
