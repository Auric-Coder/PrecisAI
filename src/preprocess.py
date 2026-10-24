"""Download CNN/DM 3.0.0 subset, tokenize, build vocab, cache tensors."""
import argparse
import os

import torch
from datasets import load_dataset

from .dataset import dirs
from .vocab import EOS_ID, PAD_ID, SOS_ID, Vocab, tokenize

SRC_LEN, TGT_LEN = 400, 60


def encode_split(ds, vocab, keep_raw):
    N = len(ds)
    src = torch.full((N, SRC_LEN), PAD_ID, dtype=torch.long)
    tgt_in = torch.full((N, TGT_LEN + 1), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((N, TGT_LEN + 1), PAD_ID, dtype=torch.long)
    src_len, tgt_len = torch.zeros(N, dtype=torch.long), torch.zeros(N, dtype=torch.long)
    for i, ex in enumerate(ds):
        s = vocab.encode(tokenize(ex["article"])[:SRC_LEN]) or [PAD_ID]
        t = vocab.encode(tokenize(ex["highlights"])[:TGT_LEN])
        src[i, : len(s)] = torch.tensor(s)
        src_len[i] = len(s)
        tgt_in[i, : len(t) + 1] = torch.tensor([SOS_ID] + t)
        tgt_out[i, : len(t) + 1] = torch.tensor(t + [EOS_ID])
        tgt_len[i] = len(t) + 1
    d = dict(src=src, src_len=src_len, tgt_in=tgt_in, tgt_out=tgt_out, tgt_len=tgt_len)
    if keep_raw:
        d["article"] = list(ds["article"])
        d["ref"] = list(ds["highlights"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--vocab_size", type=int, default=30000)
    a = ap.parse_args()
    if a.smoke:
        a.n_train, a.n_val, a.n_test = 500, 100, 50
    cache, _, _ = dirs(a.smoke)
    os.makedirs(cache, exist_ok=True)
    raw = load_dataset("abisee/cnn_dailymail", "3.0.0")
    sub = {
        "train": raw["train"].shuffle(seed=42).select(range(a.n_train)),
        "val": raw["validation"].shuffle(seed=42).select(range(a.n_val)),
        "test": raw["test"].shuffle(seed=42).select(range(a.n_test)),
    }
    vocab = Vocab.build(
        (tokenize(e["article"])[:SRC_LEN] + tokenize(e["highlights"])[:TGT_LEN] for e in sub["train"]),
        a.vocab_size,
    )
    vocab.save(os.path.join(cache, "vocab.json"))
    for name, ds in sub.items():
        torch.save(encode_split(ds, vocab, keep_raw=name != "train"), os.path.join(cache, f"{name}.pt"))
        print(f"{name}: {len(ds)} cached")
    print(f"vocab {len(vocab)} -> {cache}")


if __name__ == "__main__":
    main()
