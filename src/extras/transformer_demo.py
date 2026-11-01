"""Unit V demo: scaled dot-product attention, multi-head self-attention and sinusoidal positional encoding from scratch,
shape/numerics tests, and a BERT self-attention heatmap for a gov-doc sentence. Run: python -m src.extras.transformer_demo"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(q k^T / sqrt(d_k)) v ;  q,k,v: [..., T, d]  ->  (out [..., Tq, dv], weights [..., Tq, Tk])."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = F.softmax(scores, -1)
    return w @ v, w


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv, self.o = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):  # x [B,T,D]
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))  # [B,h,T,dk]
        out, w = scaled_dot_product_attention(q, k, v, mask)
        return self.o(out.transpose(1, 2).reshape(B, T, D)), w  # w [B,h,T,T]


def sinusoidal_positional_encoding(T, d):
    pos, i = torch.arange(T).unsqueeze(1), torch.arange(0, d, 2)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / 10000 ** (i / d)), torch.cos(pos / 10000 ** (i / d))
    return pe


def tests():
    torch.manual_seed(0)
    B, T, D, H = 2, 7, 64, 8
    x = torch.randn(B, T, D)
    out, w = scaled_dot_product_attention(x, x, x)
    assert out.shape == (B, T, D) and w.shape == (B, T, T) and torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5)
    ref = F.scaled_dot_product_attention(x, x, x)  # PyTorch reference
    assert torch.allclose(out, ref, atol=1e-5), "mismatch vs torch SDPA"
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.allclose(scaled_dot_product_attention(x, x, x, causal)[1].triu(1), torch.zeros(B, T, T))  # no attention to the future
    mha = MultiHeadSelfAttention(D, H)
    y, wh = mha(x)
    assert y.shape == (B, T, D) and wh.shape == (B, H, T, T)
    pe = sinusoidal_positional_encoding(50, D)
    assert pe.shape == (50, D) and pe.abs().max() <= 1 and torch.allclose(pe[0, 1::2], torch.ones(D // 2))  # cos(0)=1
    print("all shape/numerics tests passed")


def bert_heatmap(path, layer=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModel, AutoTokenizer
    from .common import gov_docs
    sent = next(s for d in gov_docs().values() for s in d.replace("\n", " ").split(". ")
                if 60 < len(s) < 140 and "*" not in s and sum(c.isalpha() or c == " " for c in s) > 0.95 * len(s)) + "."
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager").eval()
    x = tok(sent, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        att = bert(**x, output_attentions=True).attentions[layer][0].mean(0)  # head-averaged [T,T]
    toks = tok.convert_ids_to_tokens(x.input_ids[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(att.numpy(), cmap="viridis")
    ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=7); ax.set_yticks(range(len(toks)), toks, fontsize=7)
    ax.set_title(f"BERT layer {layer + 1} (of 12), head-averaged self-attention\n{sent[:90]}...", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print("saved", path)


if __name__ == "__main__":
    from .common import OUT
    tests()
    bert_heatmap(OUT / "bert_attention.png")
