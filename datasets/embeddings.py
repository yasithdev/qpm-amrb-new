import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class NDArrayDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        targets: np.ndarray,
        transform = None,
        target_transform = None,
    ) -> None:
        super().__init__()
        self.values = values
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        value = self.values[index]
        target = self.targets[index]

        if self.transform:
            value = self.transform(value)

        if self.target_transform:
            target = self.target_transform(target)
        return value, target

    def __len__(self):
        return self.targets.shape[0]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        embedding_path: str,
        num_dims: int,
        num_targets: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.N = 4
        self.embedding_path = embedding_path
        self.num_dims = num_dims
        num_targets = num_targets
        self.batch_size = batch_size
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str) -> None:
        if not self.dataset:
            data = np.load(self.embedding_path)
            assert "z" in data
            assert "s" in data
            self.dataset = NDArrayDataset(data["z"], data["s"])
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
