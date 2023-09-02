import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from . import NDArrayDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        embedding_path: str,
        batch_size: int,
        target_transform=None,
    ) -> None:
        super().__init__()

        self.N = 4
        self.embedding_path = embedding_path
        self.batch_size = batch_size
        self.target_transform = target_transform

        data: dict[str, np.ndarray] = np.load(self.embedding_path)
        self.dataset = NDArrayDataset(
            values=data["z"],
            targets=data["s"],
            target_transform=self.target_transform,
        )
        self.shape = self.dataset.shape
        self.labels = self.dataset.labels

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str) -> None:
        if (self.train_data or self.val_data or self.test_data) is None:
            splits = random_split(self.dataset, [0.6, 0.2, 0.2], torch.manual_seed(42))
            self.train_data, self.val_data, self.test_data = splits

    def train_dataloader(self):
        assert self.train_data
        return DataLoader(self.train_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        assert self.val_data
        return DataLoader(self.val_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        assert self.test_data
        return DataLoader(self.test_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)
