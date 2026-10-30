import os
import json
from pathlib import Path

import streamlit as st
import torch

from src.decode import beam_search, greedy, load_model, plot_attention
from src.textrank import textrank
from src.vocab import tokenize
from src.gov_eval import factual_audit

SRC_LEN = 400
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model(path):
    return load_model(path, dev)


st.set_page_config(page_title="PrecisAI", layout="wide")
st.title("PrecisAI")
mode = st.sidebar.radio("Evaluation mode", ["CNN/DailyMail demo", "Government document"])
if mode == "Government document":
    root = Path(__file__).resolve().parent
    gov_dir = root / "data" / "circulars"
    try:
        records = [json.loads(line) for line in (gov_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        records = []
    docs = [r for r in records if r.get("extraction_status") == "success" and r.get("cleaned_file")]
    labels = {"Custom / paste or upload": None}
    labels.update({f"{d['title']} [{d['document_id']}]": d for d in docs})
    selected = st.selectbox("Source metadata (optional)", list(labels))
    record = labels[selected]
    base_text = (gov_dir / record["cleaned_file"]).read_text(encoding="utf-8") if record else ""
    if record:
        st.caption(f"{record['document_type']} · {record['issuing_ministry']} · {record['publication_date']} · {record['original_url']}")
    text = st.text_area("Paste government document text", value=base_text, height=260, key="government_text")
    up = st.file_uploader("...or upload a .txt", type="txt", key="government_upload")
    if up is not None:
        text = up.read().decode("utf-8", errors="replace")
        record = None
    gov_model = st.sidebar.selectbox("Summarizer", ["TextRank", "BiGRU-GRU"])
    decode = "greedy"
    attn = None
    if gov_model == "BiGRU-GRU":
        ckpts = [m for m in ["bahdanau", "luong", "none"] if os.path.exists(f"checkpoints/{m}.pt")]
        if not ckpts:
            st.sidebar.error("No checkpoints in checkpoints/. Run src.train first.")
        else:
            attn = st.sidebar.selectbox("Checkpoint", ckpts)
            decode = st.sidebar.radio("Decoding", ["greedy", "beam"], format_func=lambda x: "Greedy" if x == "greedy" else "Beam search (4)")
    if st.button("Summarize government document", type="primary", key="gov_summarize") and text.strip():
        source_toks = tokenize(text)
        was_truncated = len(source_toks) > SRC_LEN
        attention, out_toks = None, None
        if gov_model == "TextRank":
            summary = " ".join(textrank(text))
            summary = " ".join(tokenize(summary)[:60])
        elif attn is not None:
            model, vocab = get_model(f"checkpoints/{attn}.pt")
            src_toks = source_toks[:SRC_LEN]
            src = torch.tensor([vocab.encode(src_toks) or [0]], device=dev)
            sl = torch.tensor([src.size(1)], device=dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                if decode == "greedy":
                    ids, a = greedy(model, src, sl)
                    ids = ids[0]
                    attention = a[0] if a is not None else None
                else:
                    ids, attention = beam_search(model, src, sl)
            summary = vocab.decode(ids)
            out_toks = [vocab.itos[i] for i in ids]
        else:
            st.stop()
        st.subheader("Generated summary (not an official government summary)")
        st.write(summary)
        n_src = len(source_toks)
        n_sum = max(len(tokenize(summary)), 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Source tokens", n_src)
        c2.metric("Summary tokens", n_sum)
        c3.metric("Compression ratio", f"{n_src / n_sum:.1f}x")
        if was_truncated:
            st.warning(f"Input exceeds the {SRC_LEN}-token model limit. {len(source_toks) - SRC_LEN} tokens were omitted from model input.")
        audit = factual_audit(text, summary)
        st.subheader("Policy preservation heuristic flags")
        st.caption("Automated lexical checks only; they do not prove factual correctness or clause preservation.")
        for name, result in audit.items():
            if result["unretained_items"]:
                st.warning(f"{name}: source items not found verbatim in summary: {', '.join(result['unretained_items'])}")
            else:
                st.success(f"{name}: no unmatched literal items detected")
        if attention is not None and out_toks:
            st.subheader("Attention weights")
            st.pyplot(plot_attention(attention, src_toks, out_toks))
    st.stop()

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
