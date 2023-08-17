from functools import partial
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import TileChannels2d, reindex_for_ood, take_splits
from .qpm import bacteria_dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list,
        target_label: str,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.batch_size = batch_size
        self.ood = ood
        self.target_label = target_label
        self.trainset = None
        self.valset = None
        self.testset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None
        self.transform = Compose(
            [
                ToTensor(),
                # Resize((224, 224), antialias=True),  # type: ignore
                # TileChannels2d(3),
            ]
        )
        self.shape = (1, 60, 60)
        # define targets
        strains = [
            "Acinetobacter",
            "B_subtilis",
            "E_coli_K12",
            "S_aureus",
            "E_coli_CCUG_17620",
            "E_coli_NCTC_13441",
            "E_coli_A2-39",
            "K_pneumoniae_A2-23",
            "S_aureus_CCUG_35600",
            "E_coli_101",
            "E_coli_102",
            "E_coli_104",
            "K_pneumoniae_210",
            "K_pneumoniae_211",
            "K_pneumoniae_212",
            "K_pneumoniae_240",
            "Acinetobacter_K12-21",
            "Acinetobacter_K48-42",
            "Acinetobacter_K55-13",
            "Acinetobacter_K57-06",
            "Acinetobacter_K71-71",
        ]
        species = [
            "Acinetobacter",
            "B subtilis",
            "E. coli",
            "K. pneumoniae",
            "S. aureus",
        ]
        if target_label == "species":
            self.targets = species
        elif target_label == "strain":
            self.targets = strains
        else:
            raise ValueError()
        self.permuted_targets = reindex_for_ood(self.targets, self.ood)
        self.target_transform = Compose([self.targets.__getitem__, self.permuted_targets.index])

    def get_label_splits(self):
        ind_targets = [x for i, x in enumerate(self.targets) if i not in self.ood]
        ood_targets = [x for i, x in enumerate(self.targets) if i in self.ood]
        return ind_targets, ood_targets

    def setup(self, stage: str) -> None:
        get_dataset = partial(
            bacteria_dataset,
            data_dir=self.data_root,
            transform=self.transform,
            target_transform=self.target_transform,
            label_type=self.target_label,
            balance_data=False,
        )
        if not self.trainset and not self.testset and not self.valset:
            self.trainset = get_dataset(type_="train")
            self.valset = get_dataset(type_="val")
            self.testset = get_dataset(type_="test")
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
