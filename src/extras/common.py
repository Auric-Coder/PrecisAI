"""Shared helpers for the extras (read-only use of the core pipeline)."""
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluate import rouge
from ..vocab import tokenize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "extras"
OUT.mkdir(parents=True, exist_ok=True)


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
    bad, tot = unsupported_counts(sources, summaries)
    return round(bad / tot, 4) if tot else float("nan")


def fmt_counts(sources, summaries):
    """'unsupported/total' numbers, for small samples where a bare rate hides the denominator."""
    bad, tot = unsupported_counts(sources, summaries)
    return f"{bad}/{tot}"


def distinct(preds, n):
    grams = [tuple(w[i:i + n]) for w in map(str.split, preds) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def bleu4(preds, refs):
    import sacrebleu
    return round(sacrebleu.corpus_bleu(preds, [refs], lowercase=True).score, 2)


def tri_repeat(preds):
    toks = [p.split() for p in preds]
    return float(np.mean([len(set(zip(t, t[1:], t[2:]))) < max(len(t) - 2, 0) for t in toks]))


def summary_stats(preds, refs, arts):
    return {**rouge(preds, refs), "bleu4": bleu4(preds, refs),
            "avg_len": round(float(np.mean([len(p.split()) for p in preds])), 1),
            "tri_repeat": round(tri_repeat(preds), 3),
            "distinct1": round(distinct(preds, 1), 3), "distinct2": round(distinct(preds, 2), 3),
            "unsup_num": unsupported_rate(arts, preds)}


def pagerank_select(sents, S, k=3, d=0.85, iters=100, tol=1e-6):
    """Top-k sentences (document order) by PageRank on similarity matrix S (same iteration as src/textrank.py)."""
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, 0)
    S = np.clip(S, 0, None)
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


def embedding_textrank(text, sent_vec, k=3):
    """TextRank with cosine similarity of sent_vec(sentence) vectors; zero vectors get no edges."""
    from ..textrank import split_sentences
    sents = [s.strip() for s in split_sentences(text) if s.strip()]
    if len(sents) <= k:
        return sents
    V = np.stack([sent_vec(s) for s in sents])
    V = V / np.where((n := np.linalg.norm(V, axis=1, keepdims=True)) > 0, n, 1)
    return pagerank_select(sents, V @ V.T, k)
