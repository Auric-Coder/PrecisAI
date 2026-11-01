"""T5: pretrained / fine-tuned t5-small and BERT-embedding TextRank (Unit V).
Stages: zero (t5-small inference), bert (BERT TextRank), ft (fine-tune on the same 40k subset), table (merge everything).
Predictions are scored in the project's tokenised-lowercase form so metrics match the BiGRU rows."""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration

from ..vocab import tokenize
from .common import ROOT, OUT, embedding_textrank, fmt_counts, gov_docs, load_test, summary_stats

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_LEN, TGT_LEN, SEED = 400, 60, 0
CKPT = ROOT / "checkpoints" / "extras" / "t5_small_cnndm"


def proj(p):
    return " ".join(tokenize(p))


def evaluate(name, gen, arts, refs, gov):
    """gen(list[str]) -> list[str]. Saves CNN/DM metrics + gov stats to results/extras/hf_<name>.json."""
    t0 = time.time()
    preds = [proj(p) for p in gen(arts)]
    row = summary_stats(preds, refs, arts)
    row["decode_s"] = round(time.time() - t0)
    gdocs = list(gov.values())
    gp = [proj(p) for p in gen(gdocs)]
    glen = [len(p.split()) for p in gp]
    comp = [min(len(tokenize(d)), SRC_LEN) / max(n, 1) for d, n in zip(gdocs, glen)]
    out = {"cnndm": row, "gov": {"unsup_num": fmt_counts(gdocs, gp), "avg_len": round(float(np.mean(glen)), 2),
                                 "avg_compression": round(float(np.mean(comp)), 4)}, "gov_summaries": gp}
    print(name, row, out["gov"], flush=True)
    (OUT / f"hf_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def t5_generator(model, tok, prefix="summarize: ", bs=16):
    model.eval().to(DEV)

    @torch.no_grad()
    def gen(texts):
        out = []
        for i in range(0, len(texts), bs):
            x = tok([prefix + t for t in texts[i:i + bs]], max_length=SRC_LEN, truncation=True, padding=True, return_tensors="pt").to(DEV)
            ids = model.generate(**x, num_beams=4, max_new_tokens=TGT_LEN, early_stopping=True)
            out += tok.batch_decode(ids, skip_special_tokens=True)
        return out
    return gen


def bert_vec_fn():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(DEV).eval()

    @torch.no_grad()
    def vec(s):
        x = tok(s, max_length=128, truncation=True, return_tensors="pt").to(DEV)
        h = bert(**x).last_hidden_state[0]
        return h.mean(0).cpu().numpy()  # mean-pooled over tokens (incl. [CLS]/[SEP])
    return vec


def finetune(bs=16, lr=3e-4):
    from datasets import load_dataset
    torch.manual_seed(SEED)
    tr = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train").shuffle(seed=42).select(range(40000))  # same subset as the BiGRU models
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    order = np.random.RandomState(SEED).permutation(len(tr))
    t0, run = time.time(), 0.0
    for step, i in enumerate(range(0, len(order), bs), 1):
        b = tr[[int(j) for j in order[i:i + bs]]]
        x = tok(["summarize: " + a for a in b["article"]], max_length=SRC_LEN, truncation=True, padding=True, return_tensors="pt").to(DEV)
        y = tok(b["highlights"], max_length=TGT_LEN, truncation=True, padding=True, return_tensors="pt").input_ids.to(DEV)
        y[y == tok.pad_token_id] = -100
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            loss = model(**x, labels=y).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step}/{len(order) // bs} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
    return model, tok


def table():
    dec = json.load(open(OUT / "decoding.json"))
    gm = json.load(open(ROOT / "results/government/document_metrics.json", encoding="utf-8"))["model_metrics"]
    d = load_test()
    src_mean = float(d["src_len"].float().mean())  # mean source word tokens seen by the BiGRU (<= 400)
    gov_unsup = dec["gov_unsupported_number_rate"]
    rows = [("TextRank (TF-IDF)", dec["systems"]["textrank"], "textrank"),
            ("BiGRU Bahdanau greedy", dec["systems"]["bahdanau-greedy"], "bigru_gru_bahdanau_greedy"),
            ("BiGRU Bahdanau beam4", dec["systems"]["bahdanau-beam4"], "bigru_gru_bahdanau_beam"),
            ("BiGRU Bahdanau beam4 tuned", dec["systems"]["bahdanau-beam4-tuned"], None),
            ("BiGRU Luong beam4", dec["systems"]["luong-beam4"], "bigru_gru_luong_beam"),
            ("BiGRU no-attn beam4", dec["systems"]["none-beam4"], "bigru_gru_none_beam")]
    for name, f in [("t5-small pretrained (zero-shot, beam4)", "t5_zero"), ("BERT-embedding TextRank", "bert_textrank"), ("t5-small fine-tuned (1 epoch, beam4)", "t5_ft")]:
        if (OUT / f"hf_{f}.json").exists():
            j = json.load(open(OUT / f"hf_{f}.json"))
            rows.append((name, j["cnndm"], j))
    L = ["# Extractive / BiGRU / pretrained-transformer comparison", "",
         "CNN/DM test n=500 (ROUGE F1 x100). Compression = mean source tokens (first 400 word tokens, mean "
         f"{src_mean:.0f}) / mean summary tokens. Unsupported-number rate = share of numeric tokens in the summary absent from the source (T1 definition). "
         "Gov columns: 9 documents, no references, so no ROUGE; compression there is mean of per-document ratios (results/government definition).", "",
         "| system | R-1 | R-2 | R-L | avg len | compression | unsup-num (CNN/DM) | unsup-num (gov) | gov compression |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r, g in rows:
        if isinstance(g, dict):
            gu, gc = g["gov"]["unsup_num"], g["gov"]["avg_compression"]
        else:
            gu = gov_unsup.get(g) if g else "-"
            gc = gm[g]["avg_compression_ratio"] if g in gm else "-"
        L.append(f"| {name} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} | {r['avg_len']} | {src_mean / r['avg_len']:.2f} | {r['unsup_num']} | {gu} | {gc} |")
    L += ["", "t5-small inputs are truncated to 400 SentencePiece tokens (~300 words), BiGRU inputs to 400 words; T5 uses `summarize: ` prefix, beam 4, max 60 new tokens. "
          "T5 outputs are lowercased/tokenised with the project tokenizer before scoring so the metrics are comparable. "
          "BERT-TextRank: bert-base-uncased last-layer mean-pooled sentence vectors (<=128 tokens), cosine graph, PageRank, top-3."]
    (OUT / "hf_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["zero", "bert", "ft", "table"])
    a = ap.parse_args()
    if a.stage == "table":
        return table()
    d, gov = load_test(), gov_docs()
    arts, refs = d["article"], d["ref"]
    if a.stage == "zero":
        tok = AutoTokenizer.from_pretrained("t5-small")
        evaluate("t5_zero", t5_generator(T5ForConditionalGeneration.from_pretrained("t5-small"), tok), arts, refs, gov)
    elif a.stage == "bert":
        vec = bert_vec_fn()
        evaluate("bert_textrank", lambda ts: [" ".join(embedding_textrank(t, vec)) for t in ts], arts, refs, gov)
    else:
        model, tok = finetune()
        evaluate("t5_ft", t5_generator(model, tok), arts, refs, gov)


if __name__ == "__main__":
    main()
