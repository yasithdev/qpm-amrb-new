import json
from functools import partial

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
        source_labels: list[str],
        grouping: list[int] | None = None,
        labels: list[str] | None = None,
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
        self.shape: tuple[int, ...] = tuple(info["shape"])
        # get target mapping
        self.source_labels = source_labels
        self.grouping = grouping or list(range(len(self.source_labels)))  # by default, each target has its own group
        self.labels = labels or self.source_labels
        # ood adjustments
        self.permuted_labels = reindex_for_ood(self.labels, self.ood)
        self.permuted_grouping = list(map(self.permuted_labels.index, map(self.labels.__getitem__, self.grouping)))
        self.target_transform = self.permuted_grouping.__getitem__

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
            filter_labels=[i for i, x in enumerate(self.grouping) if x in self.ood],
        )
        if stage == "fit":
            self.train_data = create_fn(emb_type="train", filter_mode="exclude")
            self.val_data = create_fn(emb_type="val", filter_mode="exclude")
        if stage == "test":
            self.test_data = create_fn(emb_type="test", filter_mode="exclude")
        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    create_fn(emb_type="train", filter_mode="include"),
                    create_fn(emb_type="val", filter_mode="include"),
                    create_fn(emb_type="test", filter_mode="include"),
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
