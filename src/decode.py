import torch
import torch.nn.functional as F

from .model import Seq2Seq
from .vocab import EOS_ID, PAD_ID, SOS_ID, UNK_ID, Vocab


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    vocab = Vocab(ck["itos"])
    model = Seq2Seq(len(vocab), attn=ck["args"]["attn"]).to(dev)
    model.load_state_dict(ck["state"])
    return model.eval(), vocab


def _logp(model, o, t, min_len):
    lp = F.log_softmax(model.out(o).float(), -1)
    lp[:, PAD_ID] = lp[:, SOS_ID] = -1e9
    if t < min_len:
        lp[:, EOS_ID] = -1e9
    return lp


@torch.no_grad()
def greedy(model, src, lens, max_len=60, min_len=10, no_repeat_ngram=3):
    """Batched greedy decoding with optional n-gram blocking. Returns (list of id lists, attention [B,T,S] or None)."""
    st = model.encode(src, lens)
    B = src.size(0)
    h, y = st["h"], torch.full((B,), SOS_ID, dtype=torch.long, device=src.device)
    ids, atts = [], []
    generated = [[] for _ in range(B)]
    for t in range(max_len):
        o, h, a = model.step(y, h, st)
        lp = _logp(model, o, t, min_len)
        if no_repeat_ngram > 0:
            for b in range(B):
                for tok in _banned(generated[b], n=no_repeat_ngram):
                    lp[b, tok] = -1e9
        y = lp.argmax(-1)
        for b in range(B):
            generated[b].append(y[b].item())
        ids.append(y)
        if a is not None:
            atts.append(a)
    ids = torch.stack(ids, 1).tolist()
    out = [r[: r.index(EOS_ID)] if EOS_ID in r else r for r in ids]
    return out, (torch.stack(atts, 1) if atts else None)


def _banned(seq, n=3):
    """Tokens that would complete an already-seen trigram after seq."""
    if len(seq) < n - 1:
        return []
    pre = tuple(seq[-(n - 1):])
    return [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == pre]


@torch.no_grad()
def beam_search(model, src, lens, beam=4, alpha=0.6, max_len=60, min_len=10, no_repeat_trigram=True, cov_penalty=0.0):
    """Single-article beam search (src [1,S]). GNMT length penalty ((5+L)/6)^alpha.
    Returns (ids, attention [T,S] or None)."""
    dev = src.device
    st = model.encode(src, lens)
    seqs, atts = [[]], [[]]
    scores = torch.zeros(1, device=dev)
    h = st["h"]
    done = []  # (normalised score, ids, att)
    for t in range(max_len):
        k = len(seqs)
        y = torch.tensor([s[-1] if s else SOS_ID for s in seqs], device=dev)
        o, h, a = model.step(y, h, model.expand(st, torch.zeros(k, dtype=torch.long, device=dev)))
        lp = _logp(model, o, t, min_len)
        if no_repeat_trigram:
            for b, s in enumerate(seqs):
                for tok in _banned(s):
                    lp[b, tok] = -1e9
        V = lp.size(1)
        vals, idx = (scores[:, None] + lp).view(-1).topk(2 * k)
        nseq, natt, nsc, par = [], [], [], []
        for v, i in zip(vals.tolist(), idx.tolist()):
            b, tok = divmod(i, V)
            att = atts[b] + ([a[b]] if a is not None else [])
            if tok == EOS_ID:
                sc = v / (((5 + t + 1) / 6) ** alpha)
                if cov_penalty > 0 and att:
                    stacked_att = torch.stack(att)
                    cov_pen = torch.clamp(stacked_att.sum(0) - 1.0, min=0.0).sum().item()
                    sc -= cov_penalty * cov_pen
                done.append((sc, seqs[b], att))
            else:
                nseq.append(seqs[b] + [tok]); natt.append(att); nsc.append(v); par.append(b)
            if len(nseq) == beam:
                break
        if len(done) >= beam or not nseq:
            break
        seqs, atts = nseq, natt
        scores = torch.tensor(nsc, device=dev)
        h = h.index_select(0, torch.tensor(par, device=dev))
    if not done:
        done = [(sc / (((5 + max_len) / 6) ** alpha), s, at) for sc, s, at in zip(scores.tolist(), seqs, atts)]
    _, ids, att = max(done, key=lambda x: x[0])
    return ids, (torch.stack(att) if att else None)


def plot_attention(att, src_tokens, out_tokens, path=None, top=40, fontsize=7, colorbar=False):
    """Heatmap of attention [T,S]; shows the `top` most-attended source tokens (original order)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    att = att.float().cpu()[: len(out_tokens), : len(src_tokens)]
    keep = sorted(att.sum(0).topk(min(top, att.size(1))).indices.tolist())
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(keep)), max(3, 0.25 * len(out_tokens))))
    im = ax.imshow(att[:, keep].numpy(), aspect="auto", cmap="viridis")
    if colorbar:
        fig.colorbar(im, ax=ax, label="attention weight")
    ax.set_xticks(range(len(keep)), [src_tokens[i] for i in keep], rotation=90, fontsize=fontsize)
    ax.set_yticks(range(len(out_tokens)), out_tokens, fontsize=fontsize)
    ax.set_xlabel("source (top attended)"); ax.set_ylabel("generated")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig
