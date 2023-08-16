import lightning.pytorch as pt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from ..transforms import AddGaussianNoise, reindex_for_ood, take_splits
from .amrb import create_datasets, get_target_map


class DataModule(pt.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list,
        version: str,
        target_label: str,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.batch_size = batch_size
        self.ood = ood
        self.version = version
        self.target_label = target_label
        self.trainset = None
        self.testset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None
        self.shape = (1, 40, 40)
        self.transform = Compose(
            [
                ToTensor(),
                AddGaussianNoise(mean=0.0, std=0.01),
            ]
        )
        target_map = get_target_map(data_root, version, target_label)
        self.targets = sorted(target_map.keys())
        self.permuted_targets = reindex_for_ood(self.targets, self.ood)
        self.target_transform = Compose([self.targets.__getitem__, self.permuted_targets.index])

    def get_label_splits(self):
        ind_targets = [x for i, x in enumerate(self.targets) if i not in self.ood]
        ood_targets = [x for i, x in enumerate(self.targets) if i in self.ood]
        return ind_targets, ood_targets

    def setup(self, stage: str) -> None:
        if not self.trainset and not self.testset:
            self.trainset, self.testset = create_datasets(
                data_root=self.data_root,
                version=self.version,
                target_label=self.target_label,
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
