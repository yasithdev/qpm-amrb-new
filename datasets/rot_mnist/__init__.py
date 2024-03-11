from functools import partial

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import TileChannels2d, ZeroPad2D
from .rot_mnist import rot_mnist_dataset

Digits = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

rotation_labels = [
    "Rotated",
    "Unrotated",
]

digit_to_rotation_mapping = [1,1,1,1,1,0,0,0,0,0]

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list[int],
        aug_hw_224: bool = True,
        aug_ch_3: bool = True,
        target_transform = None,
        fake_mode = False,
        shuffle_training_data: bool = True,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.batch_size = batch_size
        self.ood = ood
        self.aug_hw_224 = aug_hw_224
        self.aug_ch_3 = aug_ch_3
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None
        # define targets
        self.classes = rotation_labels
        self.target_transform = target_transform
        self.fake_mode = fake_mode
        self.shuffle_training_data = shuffle_training_data

        # pre-transform shape
        self.orig_shape = (1, 28, 28)
        c, h, w = self.orig_shape

        # transform
        trans = []
        trans.append(ToTensor())
        trans.append(ZeroPad2D(2, 2, 2, 2))
        h = w = 28
        if self.aug_hw_224:
            trans.append(Resize(size=(224, 224), antialias=True))
            h = w = 224
        if self.aug_ch_3:
            trans.append(TileChannels2d(3))
            c = 3
        self.transform = Compose(trans)

        # post-transform shape
        self.shape = (c, h, w)

    def setup(self, stage: str) -> None:
        get_dataset = partial(
            rot_mnist_dataset,
            data_dir=self.data_root,
            transform=self.transform,
            target_transform=self.target_transform,
            filter_labels=self.ood,
            fake_mode = self.fake_mode
        )
        if stage == "fit":
            self.train_data = get_dataset(type_="train", filter_mode="exclude")
            self.val_data = get_dataset(type_="val", filter_mode="exclude")

        if stage == "test":
            self.test_data = get_dataset(type_="test", filter_mode="exclude")

        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    # get_dataset(type_="train", filter_mode="include"),
                    # get_dataset(type_="val", filter_mode="include"),
                    get_dataset(type_="test", filter_mode="include"),
                ]
            )

    def train_dataloader(self):
        assert self.train_data
        return DataLoader(self.train_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=self.shuffle_training_data)

    def val_dataloader(self):
        assert self.val_data
        return DataLoader(self.val_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        assert self.test_data
        return DataLoader(self.test_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)

    def predict_dataloader(self):
        assert self.ood_data
        return DataLoader(self.ood_data, self.batch_size, num_workers=self.N, pin_memory=True, shuffle=False)
