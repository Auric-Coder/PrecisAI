"""T1: sampling decoders, perplexity, BLEU-4, unsupported-number rate. Writes results/extras/decoding.{md,json}."""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..decode import _logp, beam_search, greedy, load_model
from ..textrank import textrank
from ..vocab import EOS_ID, PAD_ID, SOS_ID, tokenize
from .common import ROOT, OUT, fmt_counts, gov_docs, load_test, summary_stats

SEEDS = [0, 1, 2]


@torch.no_grad()
def sample(model, src, lens, mode, temp=1.0, k=40, p=0.9, max_len=60, min_len=10, gen=None):
    """Batched ancestral sampling with temperature and top-k / nucleus truncation."""
    st = model.encode(src, lens)
    h, y = st["h"], torch.full((src.size(0),), SOS_ID, dtype=torch.long, device=src.device)
    ids = []
    for t in range(max_len):
        o, h, _ = model.step(y, h, st)
        lp = F.log_softmax(_logp(model, o, t, min_len) / temp, -1)
        if mode == "topk":
            kth = lp.topk(k, -1).values[:, -1:]
            lp = lp.masked_fill(lp < kth, -1e9)
        elif mode == "nucleus":
            sp, si = lp.exp().sort(-1, descending=True)
            drop = (sp.cumsum(-1) - sp) >= p  # keep tokens until mass >= p (first crossing kept)
            lp = lp.masked_fill(torch.zeros_like(drop).scatter(1, si, drop), -1e9)
        y = torch.multinomial(F.softmax(lp, -1), 1, generator=gen).squeeze(1)
        ids.append(y)
    out = torch.stack(ids, 1).tolist()
    return [r[: r.index(EOS_ID)] if EOS_ID in r else r for r in out]


@torch.no_grad()
def perplexity(model, d, dev, bs=100):
    tot = n = 0
    for i in range(0, len(d["src"]), bs):
        sl = d["src_len"][i:i + bs].to(dev)
        tl = d["tgt_len"][i:i + bs].max()
        src, ti, to = d["src"][i:i + bs, : sl.max()].to(dev), d["tgt_in"][i:i + bs, :tl].to(dev), d["tgt_out"][i:i + bs, :tl].to(dev)
        loss = F.cross_entropy(model(src, sl, ti).float().flatten(0, 1), to.flatten(), ignore_index=PAD_ID, reduction="sum")
        tot += loss.item(); n += (to != PAD_ID).sum().item()
    return math.exp(tot / n)


def batches(model, vocab, d, dev, fn, bs=100):
    out = []
    for i in range(0, len(d["src"]), bs):
        sl = d["src_len"][i:i + bs].to(dev)
        out += [vocab.decode(x) for x in fn(model, d["src"][i:i + bs, : sl.max()].to(dev), sl)]
    return out


def beam_all(model, vocab, d, dev, arts, unk_replace=False, **kw):
    out = []
    for i in range(len(d["src"])):
        sl = d["src_len"][i:i + 1].to(dev)
        ids, att = beam_search(model, d["src"][i:i + 1, : sl.item()].to(dev), sl, **kw)
        out.append(vocab.decode(ids, tokenize(arts[i])[:400] if unk_replace else None, att if unk_replace else None))
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_test()
    refs, arts = d["ref"], d["article"]
    rows, ppl, samp = {}, {}, {}
    gd = gov_docs()
    gsum = json.load(open(ROOT / "results/government/summaries.json", encoding="utf-8"))
    gov = {name: fmt_counts(list(gd.values()), s) for name, s in gsum.items()}  # existing gov summaries, read-only

    rows["textrank"] = summary_stats([" ".join(textrank(t)) for t in arts], refs, arts)
    for m in ["bahdanau", "luong", "none"]:
        model, vocab = load_model(str(ROOT / f"checkpoints/{m}.pt"), dev)
        ppl[m] = round(perplexity(model, d, dev), 2)
        # reproduce metrics.md rows (greedy without n-gram blocking, beam4 alpha .6 min_len 10)
        rows[f"{m}-greedy"] = summary_stats(batches(model, vocab, d, dev, lambda a, b, c: greedy(a, b, c, no_repeat_ngram=0)[0]), refs, arts)
        rows[f"{m}-beam4"] = summary_stats(beam_all(model, vocab, d, dev, arts), refs, arts)
        print(m, ppl[m], rows[f"{m}-greedy"], rows[f"{m}-beam4"], flush=True)
        if m == "bahdanau":
            rows["bahdanau-beam4-tuned"] = summary_stats(
                beam_all(model, vocab, d, dev, arts, unk_replace=True, beam=4, alpha=1.0, min_len=20), refs, arts)
            for mode, temp in [("topk", 1.0), ("topk", 0.7), ("nucleus", 1.0), ("nucleus", 0.7)]:
                runs = []
                for seed in SEEDS:
                    gen = torch.Generator(device=dev).manual_seed(seed)
                    preds = batches(model, vocab, d, dev, lambda a, b, c: sample(a, b, c, mode, temp, gen=gen))
                    runs.append(summary_stats(preds, refs, arts))
                key = f"bahdanau-{mode}-T{temp}"
                samp[key] = {c: (round(float(np.mean([r[c] for r in runs])), 3), round(float(np.std([r[c] for r in runs])), 3)) for c in runs[0]}
                print(key, samp[key], flush=True)
    json.dump(dict(perplexity=ppl, systems=rows, sampling=samp, gov_unsupported_number_rate=gov), open(OUT / "decoding.json", "w"), indent=1)

    cols = [("rouge1", "R-1"), ("rouge2", "R-2"), ("rougeL", "R-L"), ("bleu4", "BLEU-4"), ("avg_len", "len"), ("tri_repeat", "tri-rep"),
            ("distinct1", "dist-1"), ("distinct2", "dist-2"), ("unsup_num", "unsup-num")]
    hdr = ["| " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    L = ["# Decoding study (CNN/DM test, n=500)", "",
         "Generated by `python -m src.extras.decoding_study`. Core checkpoints and code untouched.", "",
         "## Deterministic systems", "", "| system " + hdr[0], hdr[1]]
    L += [f"| {k} | " + " | ".join(str(v[c]) for c, _ in cols) + " |" for k, v in rows.items()]
    L += ["", "greedy = no n-gram blocking; beam4 = alpha 0.6, min_len 10; beam4-tuned = alpha 1.0, min_len 20, attention UNK replacement.",
          "The greedy/beam4 rows should reproduce results/metrics.md (compare R-1/R-2/R-L).", "",
          "## Sampling (Bahdanau, mean ± std over seeds 0,1,2; min_len 10, max_len 60, no n-gram blocking, no UNK replacement)", "",
          "| decoder " + hdr[0], hdr[1]]
    L += [f"| {k} | " + " | ".join(f"{v[c][0]} ± {v[c][1]}" for c, _ in cols) + " |" for k, v in samp.items()]
    L += ["", "top-k: k=40; nucleus: p=0.9; T = softmax temperature.", "",
          "## Test-set perplexity (teacher-forced, per target token incl. EOS, n=500)", "", "| model | perplexity |", "|---|---|"]
    L += [f"| {m} | {p} |" for m, p in ppl.items()]
    L += ["", "## Unsupported-number rate (hallucination proxy)", "",
          "Share of purely-numeric tokens (project tokenizer, so `1,000` = `1`,`000`) in the summary absent from the full source document, "
          "pooled over all summaries. Lexical proxy: a number can be supported yet misused; number words are ignored.", "",
          "**CNN/DM**: `unsup-num` columns above.", "",
          "**Government documents** (9 docs, existing `results/government/summaries.json`; each summary is checked against its full cleaned document):", "",
          "| system | unsupported / numbers in summaries |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in gov.items()]
    (OUT / "decoding.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
