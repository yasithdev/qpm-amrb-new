from functools import partial

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import TileChannels2d, ZeroPad2D, reindex_for_ood
from .qpm import bacteria_dataset, species_mapping

strains = [
    "AB",
    "BS",
    "EC_K12",
    "SA",
    "EC_CCUG_17620",
    "EC_NCTC_13441",
    "EC_A2-39",
    "KP_A2-23",
    "SA_CCUG_35600",
    "EC_101",
    "EC_102",
    "EC_104",
    "KP_210",
    "KP_211",
    "KP_212",
    "KP_240",
    "AB_K12-21",
    "AB_K48-42",
    "AB_K55-13",
    "AB_K57-06",
    "AB_K71-71",
]
species = [
    "AB",
    "BS",
    "EC",
    "KP",
    "SA",
]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list[int],
        target_label: str,
        aug_hw_224: bool = False,
        aug_ch_3: bool = True,
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
        self.strains = strains
        self.species = species
        if target_label == "species":
            self.target_labels = self.species
        elif target_label == "strain":
            self.target_labels = self.strains
        else:
            raise ValueError()
        self.permuted_labels = reindex_for_ood(self.target_labels, self.ood)
        self.target_transform = list(map(self.permuted_labels.index, self.target_labels)).__getitem__
        self.target_inv_transform = list(map(self.target_labels.index, self.permuted_labels)).__getitem__

        # pre-transform shape
        self.orig_shape = (1, 60, 60)
        c, h, w = self.orig_shape

        # transform
        trans = []
        trans.append(ToTensor())
        trans.append(ZeroPad2D(2, 2, 2, 2))
        h = w = 64
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
            bacteria_dataset,
            data_dir=self.data_root,
            transform=self.transform,
            target_transform=self.target_transform,
            label_type=self.target_label,
            balance_data=False,
        )
        if stage == "fit":
            self.train_data = get_dataset(type_="train", filter_labels=self.ood, filter_mode="exclude")

        if stage == "validate":
            self.val_data = get_dataset(type_="val", filter_labels=self.ood, filter_mode="exclude")

        if stage == "test":
            self.test_data = get_dataset(type_="test", filter_labels=self.ood, filter_mode="exclude")

        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    get_dataset(type_="train", filter_labels=self.ood, filter_mode="include"),
                    get_dataset(type_="val", filter_labels=self.ood, filter_mode="include"),
                    get_dataset(type_="test", filter_labels=self.ood, filter_mode="include"),
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
