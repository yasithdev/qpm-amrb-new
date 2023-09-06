from functools import partial
from typing import List

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from .transforms import AddGaussianNoise, TileChannels2d, ZeroPad2D, ind_ood_split, reindex_for_ood


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: List[int] = [],
        aug_ch_3: bool = False,
        add_noise: bool = True,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.ood = ood
        self.batch_size = batch_size
        self.aug_ch_3 = aug_ch_3
        self.add_noise = add_noise
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None

        # pre-transform shape
        self.orig_shape = (1, 28, 28)
        c, h, w = self.orig_shape

        # transform
        trans = []
        trans.append(ToTensor())
        trans.append(ZeroPad2D(2, 2, 2, 2))
        h = w = 32
        if self.aug_ch_3:
            trans.append(TileChannels2d(3))
            c = 3
        if self.add_noise:
            trans.append(AddGaussianNoise(mean=0.0, std=0.01))
        self.transform = Compose(trans)

        # post-transform shape
        self.shape = (c, h, w)

        # targets
        self.target_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.permuted_labels = reindex_for_ood(self.target_labels, self.ood)
        self.target_transform = list(map(self.permuted_labels.index, self.target_labels)).__getitem__
        self.target_inv_transform = list(map(self.target_labels.index, self.permuted_labels)).__getitem__

        # indexing
        create_dataset = partial(
            MNIST,
            root=self.data_root,
            download=True,
            transform=None,
            target_transform=None,
        )
        trn_id, trn_od = ind_ood_split(create_dataset(train=True), self.ood)
        tst_id, tst_od = ind_ood_split(create_dataset(train=False), self.ood)
        pivot = int(len(trn_id) * 0.8)
        trn_id, val_id = trn_id[:pivot], trn_id[pivot:]
        self.trn_id = trn_id
        self.val_id = val_id
        self.tst_id = tst_id
        self.trn_od = trn_od
        self.tst_od = tst_od

    def setup(self, stage: str) -> None:
        create_dataset = partial(
            MNIST,
            root=self.data_root,
            download=False,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        if stage == "fit":
            self.train_data = Subset(create_dataset(train=True), self.trn_id)
            self.val_data = Subset(create_dataset(train=True), self.val_id)
        if stage == "test":
            self.test_data = Subset(create_dataset(train=False), self.tst_id)
        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    Subset(create_dataset(train=True), self.trn_od),
                    Subset(create_dataset(train=False), self.tst_od),
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
