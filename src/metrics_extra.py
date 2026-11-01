"""Additional metrics: test perplexity, BLEU-4, distinct-n, unsupported-number rate (hallucination proxy)."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .vocab import PAD_ID, tokenize

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load_test():
    return torch.load(ROOT / "data" / "cache" / "test.pt", weights_only=False)


def gov_docs():
    """Cleaned gov docs in manifest order (same order as results/government/summaries.json)."""
    base = ROOT / "data" / "circulars"
    items = [json.loads(l) for l in (base / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {i["document_id"]: (base / i["cleaned_file"]).read_text(encoding="utf-8")
            for i in items if i.get("extraction_status") == "success" and i.get("cleaned_file")}


def numbers(text):
    """Purely numeric tokens under the project tokenizer (so '1,000' -> '1','000' on both sides)."""
    return [t for t in tokenize(text) if t.isdigit()]


def unsupported_counts(sources, summaries):
    bad = tot = 0
    for s, p in zip(sources, summaries):
        src = set(numbers(s))
        nums = numbers(p)
        tot += len(nums)
        bad += sum(n not in src for n in nums)
    return bad, tot


def unsupported_rate(sources, summaries):
    """Share of numbers in the summaries that never occur in the source (hallucination proxy; NaN if no numbers)."""
    bad, tot = unsupported_counts(sources, summaries)
    return round(bad / tot, 4) if tot else float("nan")


def distinct(preds, n):
    grams = [tuple(w[i:i + n]) for w in map(str.split, preds) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def bleu4(preds, refs):
    import sacrebleu
    return round(sacrebleu.corpus_bleu(preds, [refs], lowercase=True).score, 2)


def extra_columns(preds, refs, arts):
    return {"bleu4": bleu4(preds, refs), "distinct1": round(distinct(preds, 1), 3),
            "distinct2": round(distinct(preds, 2), 3), "unsup_num": unsupported_rate(arts, preds)}


@torch.no_grad()
def perplexity(model, d, dev, bs=100):
    """Teacher-forced test perplexity per target token (incl. EOS)."""
    tot = n = 0
    for i in range(0, len(d["src"]), bs):
        sl = d["src_len"][i:i + bs].to(dev)
        tl = d["tgt_len"][i:i + bs].max()
        src, ti, to = d["src"][i:i + bs, : sl.max()].to(dev), d["tgt_in"][i:i + bs, :tl].to(dev), d["tgt_out"][i:i + bs, :tl].to(dev)
        loss = F.cross_entropy(model(src, sl, ti).float().flatten(0, 1), to.flatten(), ignore_index=PAD_ID, reduction="sum")
        tot += loss.item(); n += (to != PAD_ID).sum().item()
    return round(math.exp(tot / n), 2)
