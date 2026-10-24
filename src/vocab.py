import json
import re
from collections import Counter

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
_TOK = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return _TOK.findall(text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists, size=30000):
        cnt = Counter()
        for toks in token_lists:
            cnt.update(toks)
        itos = [PAD, UNK, SOS, EOS]
        itos += [w for w, _ in cnt.most_common(size - len(itos))]
        return cls(itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, SOS_ID):
                continue
            out.append(self.itos[i])
        return " ".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
