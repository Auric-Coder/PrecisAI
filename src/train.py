import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .dataset import dirs, loader
from .model import Seq2Seq
from .vocab import PAD_ID, Vocab


def run_epoch(model, dl, dev, opt=None, scaler=None, clip=5.0):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, sl, ti, to, _ in dl:
            src, sl, ti, to = src.to(dev), sl.to(dev), ti.to(dev), to.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(src, sl, ti)
            loss = F.cross_entropy(logits.float().flatten(0, 1), to.flatten(), ignore_index=PAD_ID)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
            k = (to != PAD_ID).sum().item()
            tot += loss.item() * k
            n += k
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="bahdanau", choices=["bahdanau", "luong", "none"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 1
    cache, ckpt_dir, res_dir = dirs(a.smoke)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(os.path.join(cache, "vocab.json"))
    tr, va = loader(cache, "train", a.bs, True), loader(cache, "val", a.bs, False)
    model = Seq2Seq(len(vocab), attn=a.attn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    print(f"attn={a.attn} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M dev={dev}")
    hist, best, bad = [], 1e9, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        trl = run_epoch(model, tr, dev, opt, scaler)
        val = run_epoch(model, va, dev)
        hist.append(dict(epoch=ep, train=trl, val=val))
        print(f"ep {ep} train {trl:.3f} val {val:.3f} ppl {torch.exp(torch.tensor(val)):.1f} {time.time()-t0:.0f}s", flush=True)
        if val < best:
            best, bad = val, 0
            torch.save(dict(state=model.state_dict(), args=dict(attn=a.attn), itos=vocab.itos),
                       os.path.join(ckpt_dir, f"{a.attn}.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break
    with open(os.path.join(res_dir, f"loss_{a.attn}.json"), "w") as f:
        json.dump(hist, f)


if __name__ == "__main__":
    main()
