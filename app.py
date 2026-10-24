import os

import streamlit as st
import torch

from src.decode import beam_search, greedy, load_model, plot_attention
from src.textrank import textrank
from src.vocab import tokenize

SRC_LEN = 400
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model(path):
    return load_model(path, dev)


st.set_page_config(page_title="PrecisAI", layout="wide")
st.title("PrecisAI")
text = st.text_area("Paste text", height=220)
up = st.file_uploader("...or upload a .txt", type="txt")
if up is not None:
    text = up.read().decode("utf-8", errors="ignore")

ckpts = [m for m in ["bahdanau", "luong", "none"] if os.path.exists(f"checkpoints/{m}.pt")]
kind = st.sidebar.radio("Model", ["TextRank", "Seq2seq"])
if kind == "TextRank":
    k = st.sidebar.slider("Sentences", 1, 8, 3)
else:
    if not ckpts:
        st.sidebar.error("No checkpoints in checkpoints/. Run src.train first.")
    attn = st.sidebar.selectbox("Attention variant", ckpts)
    dec = st.sidebar.radio("Decoding", ["Greedy", "Beam (4)"])

if st.button("Summarize", type="primary") and text.strip():
    n_src = len(text.split())
    att, out_toks, src_toks = None, None, None
    if kind == "TextRank":
        summary = " ".join(textrank(text, k))
    elif ckpts:
        model, vocab = get_model(f"checkpoints/{attn}.pt")
        src_toks = tokenize(text)[:SRC_LEN]
        src = torch.tensor([vocab.encode(src_toks) or [0]], device=dev)
        sl = torch.tensor([src.size(1)], device=dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            if dec == "Greedy":
                ids, a = greedy(model, src, sl)
                ids, att = ids[0], (a[0] if a is not None else None)
            else:
                ids, att = beam_search(model, src, sl)
        summary = vocab.decode(ids)
        out_toks = [vocab.itos[i] for i in ids]
    else:
        st.stop()
    st.subheader("Summary")
    st.write(summary)
    n_sum = max(len(summary.split()), 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Source words", n_src)
    c2.metric("Summary words", n_sum)
    c3.metric("Compression ratio", f"{n_src / n_sum:.1f}x")
    if att is not None and out_toks:
        st.subheader("Attention")
        st.pyplot(plot_attention(att, src_toks, out_toks))
    elif kind == "Seq2seq":
        st.caption("No attention weights for this variant (attn=none).")
