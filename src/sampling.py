"""Stochastic decoders (top-k, nucleus) with temperature. Mirrors decode.greedy's batched interface."""
import torch
import torch.nn.functional as F

from .decode import _logp
from .vocab import EOS_ID, SOS_ID


@torch.no_grad()
def sample(model, src, lens, mode="topk", temp=1.0, k=40, p=0.9, max_len=60, min_len=10, gen=None):
    """Batched ancestral sampling; returns list of id lists (truncated at EOS)."""
    st = model.encode(src, lens)
    h, y = st["h"], torch.full((src.size(0),), SOS_ID, dtype=torch.long, device=src.device)
    ids = []
    for t in range(max_len):
        o, h, _ = model.step(y, h, st)
        lp = F.log_softmax(_logp(model, o, t, min_len) / temp, -1)
        if mode == "topk":
            lp = lp.masked_fill(lp < lp.topk(k, -1).values[:, -1:], -1e9)
        else:  # nucleus: keep smallest set with mass >= p (first crossing token kept)
            sp, si = lp.exp().sort(-1, descending=True)
            drop = (sp.cumsum(-1) - sp) >= p
            lp = lp.masked_fill(torch.zeros_like(drop).scatter(1, si, drop), -1e9)
        y = torch.multinomial(F.softmax(lp, -1), 1, generator=gen).squeeze(1)
        ids.append(y)
    out = torch.stack(ids, 1).tolist()
    return [r[: r.index(EOS_ID)] if EOS_ID in r else r for r in out]
