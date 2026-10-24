import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import PAD_ID


class Seq2Seq(nn.Module):
    """BiGRU encoder + GRU decoder. attn: bahdanau | luong | none."""

    def __init__(self, vocab_size, emb=256, enc_h=256, dec_h=512, attn="bahdanau", dropout=0.3):
        super().__init__()
        self.attn_type, self.dec_h = attn, dec_h
        enc_dim = 2 * enc_h
        self.enc_dim = enc_dim
        self.drop = nn.Dropout(dropout)
        self.eemb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.enc = nn.GRU(emb, enc_h, batch_first=True, bidirectional=True)
        self.init = nn.Linear(enc_dim, dec_h)
        self.demb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        gru_in = emb
        if attn == "bahdanau":
            self.W_s = nn.Linear(dec_h, dec_h, bias=False)
            self.W_h = nn.Linear(enc_dim, dec_h)
            self.v = nn.Linear(dec_h, 1, bias=False)
            gru_in = emb + enc_dim
        elif attn == "luong":  # "general" score
            self.W = nn.Linear(dec_h, enc_dim, bias=False)
        elif attn != "none":
            raise ValueError(attn)
        self.cell = nn.GRUCell(gru_in, dec_h)
        self.pre = nn.Linear(dec_h + (0 if attn == "none" else enc_dim), dec_h)
        self.out = nn.Linear(dec_h, vocab_size)

    def encode(self, src, lens):
        e = self.drop(self.eemb(src))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.enc(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        h = torch.tanh(self.init(torch.cat([h[0], h[1]], -1)))
        st = dict(enc=out, mask=src != PAD_ID, h=h)
        if self.attn_type == "bahdanau":
            st["keys"] = self.W_h(out)
        return st

    @staticmethod
    def expand(st, idx):
        return {k: v.index_select(0, idx) for k, v in st.items()}

    def _attend(self, scores, st):
        a = F.softmax(scores.float().masked_fill(~st["mask"], -1e9), -1)
        ctx = torch.bmm(a.unsqueeze(1).to(st["enc"].dtype), st["enc"]).squeeze(1)
        return a, ctx

    def step(self, y, h, st):
        """One decoder step -> (features [B,dec_h], new h, attention [B,S] or None)."""
        e = self.drop(self.demb(y))
        a = None
        if self.attn_type == "bahdanau":
            s = self.v(torch.tanh(st["keys"] + self.W_s(h).unsqueeze(1))).squeeze(-1)
            a, ctx = self._attend(s, st)
            h = self.cell(torch.cat([e, ctx], -1), h)
            feat = torch.cat([h, ctx], -1)
        elif self.attn_type == "luong":
            h = self.cell(e, h)
            s = torch.bmm(st["enc"], self.W(h).unsqueeze(2)).squeeze(2)
            a, ctx = self._attend(s, st)
            feat = torch.cat([h, ctx], -1)
        else:
            h = self.cell(e, h)
            feat = h
        return self.drop(torch.tanh(self.pre(feat))), h, a

    def forward(self, src, lens, tgt_in):
        st = self.encode(src, lens)
        h, outs = st["h"], []
        for t in range(tgt_in.size(1)):
            o, h, _ = self.step(tgt_in[:, t], h, st)
            outs.append(o)
        return self.out(torch.stack(outs, 1))
