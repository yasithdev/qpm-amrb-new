from __future__ import annotations

from glob import glob

import numpy as np
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        emb_dir: str,
        emb_name: str,
        emb_type: str,
        filter_labels: list[int],
        filter_mode: str = "exclude",
        transform=None,
        target_transform=None,
        target_group_fn=None,
    ) -> None:
        super().__init__()
        self.emb_dir = emb_dir
        self.emb_name = emb_name
        self.emb_type = emb_type
        self.filter_labels = filter_labels
        self.filter_mode = filter_mode
        self.transform = transform
        self.target_transform = target_transform
        self.target_group_fn = target_group_fn

        # load embeddings and targets
        embeddings = []
        targets = []
        # assuming embeddings are stored as <emb_dir>/<emb_name>/<train|val|test>_<n>.npy
        for fp in glob(f"{self.emb_dir}/{self.emb_name}/{self.emb_type}_*.npy"):
            n = int(fp.rsplit("_", maxsplit=1)[-1][:-4])
            cond1 = self.filter_mode == "exclude"
            cond2 = n in self.filter_labels
            # filtering criteria
            if cond1 == cond2:
                continue
            chunk = np.load(fp)
            embeddings.append(chunk)
            targets.append(np.full(chunk.shape[0], n))
        # store embeddings and targets as numpy arrays
        self.embeddings = np.concatenate(embeddings)
        self.targets = np.concatenate(targets)

    def __getitem__(self, index):
        value = self.embeddings[index]
        target = self.targets[index]

        if self.transform:
            value = self.transform(value)

        group = target
        if self.target_group_fn:
            group = self.target_group_fn(target)

        if self.target_transform:
            target = self.target_transform(target)

        return value, target, group

    def __len__(self):
        return self.targets.shape[0]
