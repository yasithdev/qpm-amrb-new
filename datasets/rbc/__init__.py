from functools import partial

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import AddGaussianNoise, TileChannels2d, ZeroPad2D
from .rbc import rbc_dataset

patients = [
    "220106_S1",
    "220119_S1",
    "220126_H1",
    "220126_H2",
    "220126_S1",
    "220126_S2",
    "220608_S1",
    "220623_S1",
    "220713_S1",
    "220713_S2",
    "220713_S3",
    "220720_S1",
    "220804_S1",
    "220804_S2",
    "220808_S1",
    "220808_S2",
    "220816_S1",
    "220823_S1",
    "220901_H1",
    "220901_H2",
    "220901_S1",
    "220901_S2",
    "220901_S3",
    "220914_H1",
    "220914_H2",
    "220914_S1",
    "220914_S2",
    "220921_H1",
    "220921_H2",
    "220921_S1",
    "220921_S2",
    "220930_A1",
    "220930_A2",
    "220930_A3",
    "220930_S1",
    "220930_S2",
    "220930_S3",
    "221012_A1",
    "221012_A2",
    "221012_A3",
    "221012_A4",
    "221012_S1",
    "221012_S2",
    "221012_S3",
    "221012_S4",
    "221103_H1",
    "221103_H2",
    "221103_S1",
    "221103_S2"
]

classes = [
    "Healthy Cell",
    "Sickel Cell",
]

patient_to_binary_mapping = [1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1]

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list[int],
        target_label: str,
        aug_hw_224: bool = True,
        aug_ch_3: bool = True,
        target_transform = None,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.batch_size = batch_size
        self.ood = ood
        self.target_label = target_label
        self.aug_hw_224 = aug_hw_224
        self.aug_ch_3 = aug_ch_3
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None
        # define targets
        self.classes = classes
        self.target_transform = target_transform

        # pre-transform shape
        if(self.target_label == "both"):
            self.orig_shape = (2, 301, 301)
        else:
            self.orig_shape = (1, 301, 301)
        c, h, w = self.orig_shape

        # transform
        trans = []
        trans.append(ToTensor())
        trans.append(AddGaussianNoise(0.0, 0.1)) # dequantization
        
        if self.aug_hw_224:
            trans.append(Resize(size=(224, 224), antialias=True))
            h = w = 224
        else:
            # in our experiments we are interested in a low res image (otherwise reconstruction is hard)
            trans.append(Resize(size=(64, 64), antialias=True))
            h = w = 64

        if self.aug_ch_3:
            trans.append(TileChannels2d(3))
            c = 3
        self.transform = Compose(trans)

        # post-transform shape
        self.shape = (c, h, w)

    def setup(self, stage: str) -> None:
        get_dataset = partial(
            rbc_dataset,
            data_dir=self.data_root,
            transform=self.transform,
            target_transform=self.target_transform,
            image_type=self.target_label,
            filter_labels=self.ood,
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
