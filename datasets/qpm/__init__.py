from functools import partial
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import TileChannels2d, ZeroPad2D, reindex_for_ood, take_splits, get_transforms
from .qpm import bacteria_dataset

import argparse

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list[int],
        target_label: str,
        train_type: str,
        args : argparse.Namespace,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.batch_size = batch_size
        self.ood = ood
        self.target_label = target_label
        self.train_type = train_type
        self.args = args
        self.trainset = None
        self.valset = None
        self.testset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None
        self.transform = None
        self.shape = None
        # define targets
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
        if target_label == "species":
            self.targets = species
        elif target_label == "strain":
            self.targets = strains
        else:
            raise ValueError()
        self.permuted_targets = reindex_for_ood(self.targets, self.ood)
        mapping = list(map(self.permuted_targets.index, self.targets))
        self.target_transform = mapping.__getitem__

    def get_label_splits(self):
        ind_targets = [x for i, x in enumerate(self.targets) if i not in self.ood]
        ood_targets = [x for i, x in enumerate(self.targets) if i in self.ood]
        return ind_targets, ood_targets

    def setup(self, stage: str) -> None:
        
        if(self.train_type == "self_supervised"):
            # Augmentations used for self_supervised learning objective
            self.transform = get_transforms
            self.train_transform = self.transform(args, eval = False) 
            self.val_transform   = self.transform(args, eval = True if args.train_supervised else False)
            self.test_transform  = self.transform(args, eval = True)
            self.shape = (1, 64, 64)
        else:
            self.transform = Compose(
                [
                    ToTensor(),
                    ZeroPad2D(2, 2, 2, 2),
                    # Resize((224, 224), antialias=True),  # type: ignore
                    # TileChannels2d(3),
                ]
            )
            self.train_transform = self.transform
            self.val_transform  = self.transform
            self.test_transform = self.transform
            
            self.shape = (1, 64, 64)
        
        get_dataset = partial(
            bacteria_dataset,
            data_dir=self.data_root,
            target_transform=self.target_transform,
            label_type=self.target_label,
            balance_data=False,
        )
        if not self.trainset and not self.testset and not self.valset:
            self.trainset = get_dataset(type_="train", transform = self.train_transform)
            self.valset  = get_dataset(type_="val", transform = self.val_transform)
            self.testset = get_dataset(type_="test", transform = self.test_transform)
            splits = take_splits(self.trainset, self.valset, self.testset, *self.get_label_splits())
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
