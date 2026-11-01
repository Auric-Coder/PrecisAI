"""T8 (Unit VI): blinded human-evaluation sheet and aggregation.
  python -m src.human_eval make       -> results/human_eval.csv (ratings BLANK) + human_eval_key.json (system key; do not show raters)
  python -m src.human_eval aggregate  -> results/human_eval_summary.md (mean ± std per system, from filled ratings only)
Rate each summary 1-5 for fluency, coverage (key points of the article kept) and faithfulness (nothing contradicting/absent from the article)."""
import csv
import json
import random
import sys

import numpy as np
import torch

from .decode import beam_search, load_model
from .metrics_extra import RES as OUT, ROOT, load_test
from .textrank import textrank
from .vocab import tokenize

N_DOCS, SEED = 20, 0
CSV, KEY = OUT / "human_eval.csv", OUT / "human_eval_key.json"
COLS = ["item_id", "article", "summary", "fluency", "coverage", "faithfulness"]
CRIT = COLS[3:]


def make():
    from .pretrained import t5_summarize
    d = load_test()
    arts, refs = d["article"], d["ref"]
    docs = sorted(random.Random(SEED).sample(range(len(arts)), N_DOCS))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_model(str(ROOT / "checkpoints/bahdanau.pt"), dev)
    systems = {"textrank": [], "bigru_bahdanau_beam4_tuned": [], "t5_small_pretrained": [], "reference": []}
    for i in docs:
        systems["textrank"].append(" ".join(textrank(arts[i])))
        sl = d["src_len"][i:i + 1].to(dev)
        ids, att = beam_search(model, d["src"][i:i + 1, : sl.item()].to(dev), sl, beam=4, alpha=1.0, min_len=20)
        systems["bigru_bahdanau_beam4_tuned"].append(vocab.decode(ids, tokenize(arts[i])[:400], att))
        systems["t5_small_pretrained"].append(t5_summarize([arts[i]])[0])
        systems["reference"].append(refs[i])
    items = [(doc, s, systems[s][k]) for k, doc in enumerate(docs) for s in systems]
    random.Random(SEED).shuffle(items)  # blind: row order carries no system information
    key = {}
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for n, (doc, s, text) in enumerate(items, 1):
            key[f"item{n:03d}"] = {"system": s, "test_index": doc}
            w.writerow([f"item{n:03d}", arts[doc], text, "", "", ""])
    KEY.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"wrote {len(items)} rows ({N_DOCS} docs x {len(systems)} systems) to {CSV}")


def aggregate():
    key = json.loads(KEY.read_text(encoding="utf-8"))
    by = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in CRIT:
                if r[c].strip():
                    by.setdefault(key[r["item_id"]]["system"], {}).setdefault(c, []).append(float(r[c]))
    if not by:
        sys.exit("No ratings filled in yet.")
    L = ["# Human evaluation summary", "", "Mean ± std of 1-5 ratings over rated items only (n per cell shown). Blank cells are skipped.", "",
         "| system | " + " | ".join(CRIT) + " |", "|---|" + "---|" * len(CRIT)]
    for s, cr in by.items():
        L.append(f"| {s} | " + " | ".join(f"{np.mean(cr[c]):.2f} ± {np.std(cr[c]):.2f} (n={len(cr[c])})" if c in cr else "-" for c in CRIT) + " |")
    (OUT / "human_eval_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"make": make, "aggregate": aggregate}[sys.argv[1]]()
