from typing import List

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from .transforms import AddGaussianNoise, ZeroPad2D, TileChannels2d, reindex_for_ood, take_splits


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
        self.targets = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.data_root = data_root
        self.ood = ood
        self.batch_size = batch_size
        self.aug_ch_3 = aug_ch_3
        self.add_noise = add_noise
        self.trainset = None
        self.testset = None
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
        self.permuted_targets = reindex_for_ood(self.targets, self.ood)
        mapping = list(map(self.permuted_targets.index, self.targets))
        self.target_transform = mapping.__getitem__

    def get_label_splits(self):
        ind_targets = [x for i, x in enumerate(self.targets) if i not in self.ood]
        ood_targets = [x for i, x in enumerate(self.targets) if i in self.ood]
        return ind_targets, ood_targets

    def setup(self, stage: str) -> None:
        if not self.trainset and not self.testset:
            self.trainset = MNIST(
                root=self.data_root,
                train=True,
                download=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.testset = MNIST(
                root=self.data_root,
                train=False,
                download=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            splits = take_splits(self.trainset, None, self.testset, *self.get_label_splits())
            self.train_data, self.val_data, self.test_data, self.ood_data = splits

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
