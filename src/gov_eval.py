"""Independent evaluation path for government documents and circulars.

Uses the project's existing TextRank, checkpoint loader, greedy and beam decoders.
No government document is added to the CNN/DailyMail caches or training splits.
"""
import argparse
import json
import re
from pathlib import Path

import torch

from .decode import beam_search, greedy, load_model
from .metrics_extra import unsupported_rate
from .pretrained import HF_MODELS, summarize
from .textrank import split_sentences, textrank
from .vocab import tokenize

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "circulars"
SRC_LEN, TGT_LEN = 400, 60


def load_manifest():
    path = BASE / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def words(text):
    return tokenize(text)


def chunks(text, limit=SRC_LEN):
    """Greedy sentence-aware chunks; only oversized individual sentences split by words."""
    out, cur = [], []
    for sent in split_sentences(text):
        toks = words(sent)
        if not toks:
            continue
        if len(toks) > limit:
            if cur:
                out.append(" ".join(cur)); cur = []
            out.extend(" ".join(toks[i:i + limit]) for i in range(0, len(toks), limit))
        elif len(cur) + len(toks) > limit:
            out.append(" ".join(cur)); cur = toks
        else:
            cur.extend(toks)
    if cur:
        out.append(" ".join(cur))
    return out


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else 0.0


def factual_audit(source, summary):
    """Heuristic literal-preservation indicators, not a factuality verifier."""
    rules = {
        "dates_deadlines": r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|(?:by|before|until|from)\s+\d{1,2}\s+[A-Za-z]{3,9})\b",
        "currency_amounts_percentages": r"(?:₹|Rs\.?|INR|\$|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:crore|lakh|million|billion|%|percent))",
        "eligibility_exclusions": r"\b(?:eligible|eligibility|ineligible|excluded|exclusion|provided that|subject to|not covered|only if)\b",
        "ministries_authorities": r"\b(?:Ministry of [A-Z][\w& ,()-]+|Government of India|Government|Authority|Commission|Department|Directorate|Board)\b",
        "obligations_negations_exceptions": r"\b(?:shall|must|required|obligat\w*|prohibit\w*|not|unless|except|however|provided that)\b",
        "acts_rules_notifications": r"\b(?:Act,?\s+\d{4}|Rules?,?\s+\d{4}|Notification|S\.O\.?\s*\d+\s*\(?E?\)?)\b",
    }
    result = {}
    for label, pattern in rules.items():
        src = set(re.findall(pattern, source, flags=re.I))
        pred = set(re.findall(pattern, summary, flags=re.I))
        result[label] = {"source_items": sorted(src, key=str.lower), "retained_items": sorted(pred, key=str.lower),
                         "unretained_items": sorted(src - pred, key=str.lower)}
    return result


def model_summary(text, model_name, decode, device):
    checkpoint = ROOT / "checkpoints" / f"{model_name}.pt"
    if not checkpoint.is_file():
        return None, "checkpoint_missing"
    model, vocab = load_model(str(checkpoint), device)
    toks = tokenize(text)[:SRC_LEN]
    src = torch.tensor([vocab.encode(toks) or [0]], device=device)
    lens = torch.tensor([src.size(1)], device=device)
    with torch.inference_mode(), torch.autocast("cuda", enabled=device.type == "cuda"):
        if decode == "beam":
            ids, _ = beam_search(model, src, lens)
        else:
            ids, _ = greedy(model, src, lens)
            ids = ids[0]
    return vocab.decode(ids), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["bahdanau", "luong", "none"], choices=["bahdanau", "luong", "none", *HF_MODELS])
    ap.add_argument("--decode", choices=["greedy", "beam", "both"], default="both")
    ap.add_argument("--chunking", action="store_true", help="sentence-aware chunking for documents longer than 400 tokens")
    ap.add_argument("--output", default="results/government", help="government-only result folder")
    ap.add_argument("--extra_metrics", action="store_true", help="Add unsupported-number rate; appends to results/government/metrics.md instead of overwriting")
    a = ap.parse_args()
    extended = a.extra_metrics or any(m in HF_MODELS for m in a.models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / a.output
    outdir.mkdir(parents=True, exist_ok=True)
    docs, outputs, sources = [], {}, {}
    for item in load_manifest():
        if item.get("extraction_status") != "success" or not item.get("cleaned_file"):
            continue
        source = (BASE / item["cleaned_file"]).read_text(encoding="utf-8")
        all_toks = words(source)
        over = len(all_toks) > SRC_LEN
        pieces = chunks(source) if over and a.chunking else [" ".join(all_toks[:SRC_LEN])]
        retained = len(all_toks) if a.chunking else min(len(all_toks), SRC_LEN)
        truncation = ratio(len(all_toks) - retained, len(all_toks))
        row = {"document_id": item["document_id"], "document_type": item["document_type"],
               "split": "government_evaluation",
               "original_token_count": len(all_toks), "retained_token_count": retained,
               "truncation_ratio": truncation, "exceeds_400_token_limit": over,
               "chunk_count": len(pieces) if a.chunking else None, "models": {}}
        if over:
            print(f"WARNING {item['document_id']}: {len(all_toks)} tokens; base limit {SRC_LEN}, retained {retained}, truncation {truncation:.1%}")
        for piece_id, piece in enumerate(pieces, 1):
            result = {"chunk_id": piece_id if a.chunking else None, "source_token_count": len(words(piece)), "summaries": {}}
            summary_methods = [("textrank", "extractive", None)]
            for model_name in a.models:
                if model_name in HF_MODELS:
                    summary_methods.append((model_name, "hf", model_name))
                    continue
                if a.decode in ("greedy", "both"):
                    summary_methods.append((f"bigru_gru_{model_name}_greedy", "greedy", model_name))
                if a.decode in ("beam", "both"):
                    summary_methods.append((f"bigru_gru_{model_name}_beam", "beam", model_name))
            for name, method, model_name in summary_methods:
                if method == "extractive":
                    summary = " ".join(textrank(piece))
                    status = "ok"
                elif method == "hf":
                    summary, status = summarize(model_name, [piece if a.chunking else source])[0], "ok"
                else:
                    summary, status = model_summary(piece, model_name, method, device)
                if status != "ok":
                    result["summaries"][name] = {"status": status}
                    continue
                s_words = words(summary)
                if len(s_words) > TGT_LEN:
                    summary = " ".join(s_words[:TGT_LEN])
                result["summaries"][name] = {"text": summary, "summary_token_count": len(words(summary)),
                    "compression_ratio": ratio(len(words(piece)), len(words(summary))),
                    "factual_audit_heuristics": factual_audit(piece, summary)}
                outputs.setdefault(name, []).append(summary)
                sources.setdefault(name, []).append(source)
            row["models"][str(piece_id) if a.chunking else "document"] = result
        docs.append(row)
    n_units = sum(x.get("chunk_count") or 1 for x in docs)
    aggregate = {}
    for name, preds in outputs.items():
        lens = [len(words(p)) for p in preds]
        compressions = []
        for doc in docs:
            for unit in doc["models"].values():
                summary = unit["summaries"].get(name, {})
                if "compression_ratio" in summary:
                    compressions.append(summary["compression_ratio"])
        aggregate[name] = {"n_summaries": len(preds), "avg_summary_length_tokens": round(sum(lens) / len(lens), 2) if lens else 0,
                           "avg_compression_ratio": round(sum(compressions) / len(compressions), 4) if compressions else None,
                           "rouge1_f1": None, "rouge2_f1": None, "rougeL_f1": None}
        if a.extra_metrics:
            aggregate[name]["unsupported_number_rate"] = unsupported_rate(sources[name], preds)
    if extended:  # append-only: baseline government results stay untouched
        lines = ["", f"## Extended comparison (models: {' '.join(a.models)}; chunking={a.chunking})", "",
                 "| model | summaries | avg summary tokens | avg compression ratio |" + (" unsupported-number rate |" if a.extra_metrics else ""),
                 "|---|---:|---:|---:|" + ("---:|" if a.extra_metrics else "")]
        lines += [f"| {n} | {m['n_summaries']} | {m['avg_summary_length_tokens']} | {m['avg_compression_ratio']} |"
                  + (f" {m['unsupported_number_rate']} |" if a.extra_metrics else "") for n, m in aggregate.items()]
        if a.extra_metrics:
            lines += ["", "Unsupported-number rate: share of numeric tokens in a summary absent from its full source document (NaN = summaries contain no numbers). Hallucination proxy only."]
        with open(outdir / "metrics.md", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        (outdir / "extended_summaries.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Extended government evaluation appended to {outdir / 'metrics.md'}")
        return
    report = {"n_documents": len(docs), "n_evaluation_units": n_units, "evaluation_split": "government_evaluation",
        "references_available": False, "model_metrics": aggregate, "rouge": None,
        "note": "No human-reviewed references are present; ROUGE is not computed. Compression is input tokens divided by output tokens. Factual audits are heuristic only.", "documents": docs}
    (outdir / "document_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "summaries.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"Government evaluation: {len(docs)} documents, {n_units} evaluation units", "",
        "ROUGE-1/2/L F1 not computed: no human-reviewed references are available.",
        "All documents and their chunks belong to the single government_evaluation split.", "",
        "| model | summaries | avg summary tokens | avg compression ratio | R-1 F1 | R-2 F1 | R-L F1 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {name} | {m['n_summaries']} | {m['avg_summary_length_tokens']} | {m['avg_compression_ratio']} | N/A | N/A | N/A |" for name, m in aggregate.items())
    lines += ["", "Input limit: 400 tokens per model call; summary limit: 60 tokens.",
        "Over-limit documents are flagged in document_metrics.json. With --chunking, sentence-aware chunks are evaluated separately.",
        "Policy preservation checks are lexical heuristics, not proof of factual correctness."]
    (outdir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Government evaluation complete: {len(docs)} documents; results at {outdir}")


if __name__ == "__main__":
    main()
