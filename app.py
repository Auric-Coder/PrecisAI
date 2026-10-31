import html
import json
import os
import time
from pathlib import Path

import streamlit as st
import torch

from src.decode import beam_search, greedy, load_model, plot_attention
from src.gov_eval import factual_audit
from src.textrank import split_sentences, textrank
from src.vocab import tokenize

SRC_LEN = 400
ROOT = Path(__file__).resolve().parent
GOV_DIR = ROOT / "data" / "circulars"
PASTE = "Paste your own"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(layout="wide", page_title="PrecisAI", page_icon="📝")
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: .6rem .9rem;}
.src-box {border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: .9rem 1.1rem; line-height: 1.6; max-height: 320px; overflow-y: auto;}
.src-box mark {background: rgba(255,200,0,.45); color: inherit; border-radius: 3px; padding: 0 2px;}
.chip {display: inline-block; border: 1px solid rgba(128,128,128,.4); border-radius: 999px; padding: .15rem .7rem; margin: 0 .35rem .35rem 0; font-size: .85rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(path):
    return load_model(path, dev)


@st.cache_data
def load_examples():
    try:
        lines = (GOV_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    out = {}
    for r in (json.loads(l) for l in lines if l.strip()):
        if r.get("extraction_status") == "success" and r.get("cleaned_file") and (GOV_DIR / r["cleaned_file"]).is_file():
            out[f"{r['title'][:80]} [{r['document_id']}]"] = {"meta": r, "text": (GOV_DIR / r["cleaned_file"]).read_text(encoding="utf-8")}
    return out


def run_textrank(text, k):
    t0 = time.perf_counter()
    sents = textrank(text, k)
    return {"summary": " ".join(sents), "sents": sents, "att": None, "out_toks": None, "src_toks": None, "secs": time.perf_counter() - t0}


def run_seq2seq(text, attn, decode, beam, alpha, block, max_len):
    t0 = time.perf_counter()
    model, vocab = get_model(f"checkpoints/{attn}.pt")
    src_toks = tokenize(text)[:SRC_LEN]
    src = torch.tensor([vocab.encode(src_toks) or [0]], device=dev)
    sl = torch.tensor([src.size(1)], device=dev)
    with torch.autocast("cuda", enabled=dev.type == "cuda"):
        if decode == "Greedy":
            ids, a = greedy(model, src, sl, max_len=max_len, no_repeat_ngram=3 if block else 0)
            ids, att = ids[0], (a[0] if a is not None else None)
        else:
            ids, att = beam_search(model, src, sl, beam=beam, alpha=alpha, max_len=max_len, no_repeat_trigram=block)
    return {"summary": vocab.decode(ids), "sents": None, "att": att, "out_toks": [vocab.itos[i] for i in ids],
            "src_toks": src_toks, "secs": time.perf_counter() - t0}


def highlighted_source(text, selected):
    chosen = {s.strip() for s in selected}
    parts = []
    for s in split_sentences(text):
        esc = html.escape(s.strip())
        parts.append(f"<mark>{esc}</mark>" if s.strip() in chosen else esc)
    return '<div class="src-box">' + " ".join(parts) + "</div>"


def stats(text, summary, secs):
    n_src, n_sum = len(text.split()), max(len(summary.split()), 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source words", n_src)
    c2.metric("Summary words", n_sum)
    c3.metric("Compression ratio", f"{n_src / n_sum:.1f}x", help="Source words / summary words. Higher only means shorter output.")
    c4.metric("Runtime", f"{secs * 1000:.0f} ms")


def audit_chips(text, summary):
    st.markdown("**Keyword-preservation audit**")
    chips = []
    for name, r in factual_audit(text, summary).items():
        label = name.replace("_", " ")
        if not r["source_items"]:
            chips.append(f'<span class="chip" title="Nothing of this kind in the source">➖ {label}</span>')
        elif r["unretained_items"]:
            miss = html.escape(", ".join(r["unretained_items"]), quote=True)
            chips.append(f'<span class="chip" title="Not found verbatim in summary: {miss}">⚠️ {label}</span>')
        else:
            chips.append(f'<span class="chip" title="All source items found verbatim">✅ {label}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption("Lexical checks only (hover a chip for details). They do not prove factual correctness.")


# ---------------------------------------------------------------- header + sidebar
st.title("📝 PrecisAI")
st.caption("Extractive TextRank vs. from-scratch BiGRU seq2seq summarization, with attention you can inspect.")

ckpts = [m for m in ["bahdanau", "luong", "none"] if os.path.exists(f"checkpoints/{m}.pt")]
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", ["TextRank"] + [f"Seq2seq · {m}" for m in ckpts] if ckpts else ["TextRank", "Seq2seq · bahdanau"],
                                help="TextRank is extractive (picks source sentences). Seq2seq is a BiGRU encoder-decoder; the suffix is its attention variant.")
    decode = st.radio("Decoding", ["Greedy", "Beam"], horizontal=True, help="Greedy picks the best token each step; beam search keeps several candidates.")
    beam = st.slider("Beam size", 2, 8, 4, disabled=decode == "Greedy", help="Number of candidate summaries kept during beam search.")
    alpha = st.slider("Length penalty (α)", 0.0, 1.5, 0.6, 0.1, disabled=decode == "Greedy",
                      help="Higher values favour longer beam-search outputs.")
    block = st.checkbox("Trigram blocking", value=True, help="Forbid repeating any 3-word sequence in the generated summary.")
    k = st.slider("TextRank sentences", 1, 8, 3, help="Number of sentences TextRank extracts.")
    max_len = st.slider("Seq2seq max length (tokens)", 20, 60, 60, help="Upper bound on generated summary length.")

is_neural = model_choice.startswith("Seq2seq")
attn_name = model_choice.split("· ")[-1] if is_neural else None
missing = is_neural and attn_name not in ckpts
examples = load_examples()
state = st.session_state

tab_sum, tab_cmp, tab_att, tab_about = st.tabs(["Summarize", "Compare", "Attention", "About"])

# ---------------------------------------------------------------- Summarize
with tab_sum:
    left, right = st.columns([3, 2])
    with left:
        choice = st.selectbox("Example", [PASTE] + list(examples), help="The 9 government documents in data/circulars/, or your own text.")
        if choice != PASTE:
            m = examples[choice]["meta"]
            st.caption(f"{m['document_type']} · {m['issuing_ministry']} · {m['publication_date']}")
        up = st.file_uploader("…or upload a .txt", type="txt")
        base = examples[choice]["text"] if choice != PASTE else ""
        if up is not None:
            base = up.read().decode("utf-8", errors="replace")
        text = st.text_area("Source text", value=base, height=240, key=f"text::{choice}::{up.file_id if up else ''}")
    n_words, n_toks = len(text.split()), len(tokenize(text))
    right.metric("Words", n_words, help="Live word count of the text box.")
    right.metric("Model tokens", f"{min(n_toks, SRC_LEN)} / {SRC_LEN}", help="Seq2seq models read at most 400 tokens.")
    if n_toks > SRC_LEN:
        st.warning(f"Input is {n_toks} tokens; the seq2seq models read only the first {SRC_LEN}. The rest will be truncated (TextRank uses the full text).")

    go = st.button("Summarize", type="primary")
    if go:
        if not text.strip():
            st.info("Paste some text, upload a .txt file, or pick an example first.")
        elif n_words < 30:
            st.info("That text is very short (under 30 words). Summaries work best on a full article or document.")
        elif missing:
            st.error(f"No checkpoint for '{attn_name}'. Train it first:\n\n`python -m src.train --attn {attn_name} --epochs 10`")
        else:
            with st.spinner("Generating summary…"):
                res = run_seq2seq(text, attn_name, decode, beam, alpha, block, max_len) if is_neural else run_textrank(text, k)
            state["last"] = {**res, "text": text, "model": model_choice}
    last = state.get("last")
    if last:
        st.subheader("Summary")
        st.write(last["summary"])
        st.download_button("Download summary (.txt)", last["summary"], file_name="summary.txt", mime="text/plain")
        stats(last["text"], last["summary"], last["secs"])
        if last["sents"]:
            st.markdown("**Sentences TextRank selected** (highlighted in the source)")
            st.markdown(highlighted_source(last["text"], last["sents"]), unsafe_allow_html=True)
        audit_chips(last["text"], last["summary"])
        st.caption(f"Model: {last['model']}. Generated summary, not an official one.")

# ---------------------------------------------------------------- Compare
with tab_cmp:
    st.write("Run extractive and abstractive summarizers on the text from the Summarize tab.")
    cmp_attn = attn_name if is_neural and not missing else ("bahdanau" if "bahdanau" in ckpts else (ckpts[0] if ckpts else None))
    if st.button("Compare models"):
        if not text.strip() or n_words < 30:
            st.info("Enter at least 30 words of text in the Summarize tab first.")
        elif cmp_attn is None:
            st.error("No checkpoints found. Train one first:\n\n`python -m src.train --attn bahdanau --epochs 10`")
        else:
            with st.spinner("Running both models…"):
                state["cmp"] = {"text": text, "attn": cmp_attn, "ext": run_textrank(text, k),
                                "abs": run_seq2seq(text, cmp_attn, decode, beam, alpha, block, max_len)}
    cmp = state.get("cmp")
    if cmp:
        c_ext, c_abs = st.columns(2)
        for col, title, r in [(c_ext, "Extractive · TextRank", cmp["ext"]), (c_abs, f"Abstractive · Seq2seq {cmp['attn']}", cmp["abs"])]:
            with col, st.container(border=True):
                st.subheader(title)
                st.write(r["summary"])
                n_src, n_sum = len(cmp["text"].split()), max(len(r["summary"].split()), 1)
                m1, m2, m3 = st.columns(3)
                m1.metric("Words", n_sum)
                m2.metric("Compression", f"{n_src / n_sum:.1f}x")
                m3.metric("Runtime", f"{r['secs'] * 1000:.0f} ms")

# ---------------------------------------------------------------- Attention
with tab_att:
    last = state.get("last")
    if not last or last["att"] is None:
        st.info("Summarize with a Seq2seq model that has attention (bahdanau or luong) to see its heatmap. The 'none' variant has no attention.")
    else:
        top_n = st.slider("Top-N source tokens", 10, 60, 30, help="Show only the most-attended source tokens.")
        fig = plot_attention(last["att"], last["src_toks"], last["out_toks"], top=top_n, fontsize=11, colorbar=True)
        st.pyplot(fig)

# ---------------------------------------------------------------- About
with tab_about:
    st.write(
        "PrecisAI compares unsupervised TextRank (TF-IDF cosine graph + PageRank) with a from-scratch PyTorch BiGRU encoder / GRU decoder "
        "using Bahdanau, Luong or no attention. The neural models are trained only on 40k CNN/DailyMail articles and never fine-tuned on "
        "government text, so results on the government documents are zero-shot transfer under domain shift. The 9 government documents have no "
        "reference summaries: compression ratio and the keyword audit are descriptive checks, not quality scores."
    )
