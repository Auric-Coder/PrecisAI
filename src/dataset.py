import os

import torch
from torch.utils.data import DataLoader, Dataset


def dirs(smoke=False):
    """(cache_dir, checkpoint_dir, results_dir)"""
    if smoke:
        return "data/cache_smoke", "checkpoints_smoke", "results_smoke"
    return "data/cache", "checkpoints", "results"


class SumDataset(Dataset):
    def __init__(self, cache_dir, split):
        self.d = torch.load(os.path.join(cache_dir, f"{split}.pt"), weights_only=False)

    def __len__(self):
        return len(self.d["src"])

    def __getitem__(self, i):
        d = self.d
        return d["src"][i], d["src_len"][i], d["tgt_in"][i], d["tgt_out"][i], d["tgt_len"][i]


def collate(batch):
    src, sl, ti, to, tl = map(torch.stack, zip(*batch))
    return src[:, : sl.max()], sl, ti[:, : tl.max()], to[:, : tl.max()], tl


def loader(cache_dir, split, bs, shuffle):
    return DataLoader(SumDataset(cache_dir, split), batch_size=bs, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, pin_memory=True)
