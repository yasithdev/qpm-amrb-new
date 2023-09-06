import json
from functools import partial
from typing import Callable

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader

from .common import EmbeddingDataset
from .transforms import reindex_for_ood


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        emb_dir: str,
        emb_name: str,
        batch_size: int,
        target_labels: list[str],
        group_labels: list[str] | None = None,
        target_group_fn: Callable[[int], int] | None = None,
        ood: list[int] = [],
    ) -> None:
        super().__init__()

        self.N = 4
        self.emb_dir = emb_dir
        self.emb_name = emb_name
        self.batch_size = batch_size
        self.ood = ood
        # load info
        with open(f"{self.emb_dir}/{self.emb_name}/info.json") as fp:
            info = json.load(fp)
        self.shape: tuple[int, ...] = tuple(info.shape)
        # get target mapping
        self.target_labels = target_labels
        self.group_labels = group_labels
        self.target_group_fn = target_group_fn
        self.permuted_labels = reindex_for_ood(self.target_labels, self.ood)
        self.target_transform = list(map(self.permuted_labels.index, self.target_labels)).__getitem__
        self.target_inv_transform = list(map(self.target_labels.index, self.permuted_labels)).__getitem__

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None

    def setup(self, stage: str) -> None:
        create_fn = partial(
            EmbeddingDataset,
            emb_dir=self.emb_dir,
            emb_name=self.emb_name,
            target_transform=self.target_transform,
        )

        if stage == "fit":
            self.train_data = create_fn(emb_type="train", filter_labels=self.ood, filter_mode="exclude")

        if stage == "validate":
            self.val_data = create_fn(emb_type="val", filter_labels=self.ood, filter_mode="exclude")

        if stage == "test":
            self.test_data = create_fn(emb_type="test", filter_labels=self.ood, filter_mode="exclude")

        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    create_fn(emb_type="train", filter_labels=self.ood, filter_mode="include"),
                    create_fn(emb_type="val", filter_labels=self.ood, filter_mode="include"),
                    create_fn(emb_type="test", filter_labels=self.ood, filter_mode="include"),
                ]
            )

    def train_dataloader(self):
        assert self.train_data
        return DataLoader(self.train_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        assert self.val_data
        return DataLoader(self.val_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        assert self.test_data
        return DataLoader(self.test_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)

    def predict_dataloader(self):
        assert self.ood_data
        return DataLoader(self.ood_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)
