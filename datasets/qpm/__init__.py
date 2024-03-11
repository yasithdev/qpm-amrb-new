from functools import partial

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import TileChannels2d, ZeroPad2D
from .qpm import bacteria_dataset

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
### Species level mapping
# 0 => Acinetobacter
# 1 => B. subtilis
# 2 => E. coli
# 3 => K. pneumoniae
# 4 => S. aureus
# More info => https://ruhsoft-my.sharepoint.com/:p:/g/personal/im_ramith_fyi/EYMDb528EVlClCp2y8nIM8oB9LBZ-lbqEiCXwcAZHX7wew?e=lAROoR
species_mapping = [0, 1, 2, 4, 2, 2, 2, 3, 4, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0]


# Strainwise split configuration without B. subtilis
strains_train = strains_val = [
    "AB",
    "AB_K12-21",
    "AB_K48-42",
    "AB_K55-13",
    "EC_101",
    "EC_102",
    "EC_104",
    "KP_A2-23",
    "EC_A2-39",
    "EC_CCUG_17620",
    "KP_210",
    "KP_211",
    "SA",
]
strains_test = [
    "AB_K57-06",
    "AB_K71-71",
    "EC_K12",
    "EC_NCTC_13441",
    "KP_212",
    "KP_240",
    "SA_CCUG_35600",
]
species_alt = [
    "AB",
    "EC",
    "KP",
    "SA",
]
### Species level mapping
# 0 => Acinetobacter
# 1 => E. coli
# 2 => K. pneumoniae
# 3 => S. aureus
ß = None
species_mapping_alt = [0, ß, 1, 3, 1, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list[int],
        target_label: str,
        aug_hw_224: bool = False,
        aug_ch_3: bool = True,
        target_transform=None,
        strainwise_split=False,
        shuffle_training_data: bool = True,
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
        self.target_transform = target_transform
        self.strainwise_split = strainwise_split
        self.shuffle_training_data = shuffle_training_data

        # pre-transform shape
        self.orig_shape = (1, 60, 60)
        c, h, w = self.orig_shape

        # transform
        trans = []
        trans.append(ToTensor())
        trans.append(ZeroPad2D(2, 2, 2, 2))
        h = w = 64
        if self.aug_hw_224:
            trans.append(Resize(size=(224, 224), antialias=True))  # type: ignore
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
            filter_labels=self.ood,
            strainwise_split=self.strainwise_split,
        )

        if self.strainwise_split:
            allowed_trn = list(map(strains.index, strains_train))
            allowed_val = list(map(strains.index, strains_val))
            allowed_tst = list(map(strains.index, strains_test))
        else:
            allowed_trn = allowed_val = allowed_tst = list(range(len(strains)))

        if stage == "fit":
            self.train_data = get_dataset(type_="train", filter_mode="exclude", strains=allowed_trn)
            self.val_data = get_dataset(type_="val", filter_mode="exclude", strains=allowed_val)

        if stage == "test":
            self.test_data = get_dataset(type_="test", filter_mode="exclude", strains=allowed_tst, balance_data=True)

        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    # get_dataset(type_="train", filter_mode="include", strains=allowed_trn, balance_data=True),
                    # get_dataset(type_="val", filter_mode="include", strains=allowed_val, balance_data=True),
                    get_dataset(type_="test", filter_mode="include", strains=allowed_tst, balance_data=True),
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
