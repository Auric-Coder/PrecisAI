"""Pretrained transformer baselines (inference only): t5-small abstractive and BERT-embedding TextRank.
Hooked into evaluate.py / gov_eval.py via --models t5-small bert-extractive."""
import torch

from .textrank import textrank

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODELS = ("t5-small", "bert-extractive")
SRC_TOKENS, TGT_TOKENS = 400, 60
_CACHE = {}


def t5_summarize(texts, bs=16, beams=4):
    """t5-small with the 'summarize: ' prefix; input truncated to 400 SentencePiece tokens, beam 4, <=60 new tokens."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    if "t5" not in _CACHE:
        _CACHE["t5"] = (AutoTokenizer.from_pretrained("t5-small"), T5ForConditionalGeneration.from_pretrained("t5-small").to(DEV).eval())
    tok, model = _CACHE["t5"]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            x = tok(["summarize: " + t for t in texts[i:i + bs]], max_length=SRC_TOKENS, truncation=True, padding=True, return_tensors="pt").to(DEV)
            ids = model.generate(**x, num_beams=beams, max_new_tokens=TGT_TOKENS, early_stopping=True)
            out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def bert_vec_fn():
    """sentence -> bert-base-uncased last-layer hidden states mean-pooled over tokens (<=128)."""
    if "bert" not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        _CACHE["bert"] = (AutoTokenizer.from_pretrained("bert-base-uncased"), AutoModel.from_pretrained("bert-base-uncased").to(DEV).eval())
    tok, bert = _CACHE["bert"]

    @torch.no_grad()
    def vec(s):
        x = tok(s, max_length=128, truncation=True, return_tensors="pt").to(DEV)
        return bert(**x).last_hidden_state[0].mean(0).cpu().numpy()
    return vec


def summarize(model_name, texts):
    """Summaries (raw text) for either selectable model."""
    if model_name == "t5-small":
        return t5_summarize(texts)
    vec = bert_vec_fn()
    return [" ".join(textrank(t, repr="bert", vec_fn=vec)) for t in texts]
