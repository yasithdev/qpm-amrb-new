from functools import partial

import lightning.pytorch as pt
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from ..transforms import AddGaussianNoise, TileChannels2d, reindex_for_ood
from .amrb import create_dataset, get_target_map

labels = [] # TODO fix this

class DataModule(pt.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        ood: list,
        version: str,
        target_label: str,
        aug_hw_224: bool = False,
        aug_ch_3: bool = False,
        add_noise: bool = True,
        shuffle_training_data: bool = True,
    ) -> None:
        super().__init__()
        self.N = 4
        self.data_root = data_root
        self.batch_size = batch_size
        self.ood = ood
        self.version = version
        self.target_label = target_label
        self.aug_hw_224 = aug_hw_224
        self.aug_ch_3 = aug_ch_3
        self.add_noise = add_noise
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ood_data = None
        self.shuffle_training_data = shuffle_training_data

        # pre-transform shape
        self.orig_shape = (1, 40, 40)
        c, h, w = self.orig_shape

        # transform
        trans = []
        trans.append(ToTensor())
        if self.aug_hw_224:
            trans.append(Resize((224, 224), antialias=True))
            h = w = 224
        if self.aug_ch_3:
            trans.append(TileChannels2d(3))
            c = 3
        if self.add_noise:
            trans.append(AddGaussianNoise(mean=0.0, std=0.01))
        self.transform = Compose(trans)

        # post-transform shape
        self.shape = (c, h, w)

        # targets
        target_map = get_target_map(data_root, version, target_label)
        self.target_labels = sorted(target_map.keys())
        self.permuted_labels = reindex_for_ood(self.target_labels, self.ood)
        self.target_transform = list(map(self.permuted_labels.index, self.target_labels)).__getitem__
        self.target_inv_transform = list(map(self.target_labels.index, self.permuted_labels)).__getitem__

    def setup(self, stage: str) -> None:
        create_fn = partial(
            create_dataset,
            data_root=self.data_root,
            version=self.version,
            target_label=self.target_label,
            splits=[0.6, 0.2, 0.2],
            balance_data=False,
            transform=self.transform,
            target_transform=self.target_transform,
            filter_labels=self.ood, 
        )
        if stage == "fit":
            self.train_data = create_fn(split_id=0, filter_mode="exclude")
            self.val_data = create_fn(split_id=1, filter_mode="exclude")

        if stage == "test":
            self.test_data = create_fn(split_id=2, filter_mode="exclude")

        if stage == "predict":
            self.ood_data = ConcatDataset(
                [
                    # create_fn(split_id=0, filter_mode="include"),
                    # create_fn(split_id=1, filter_mode="include"),
                    create_fn(split_id=2, filter_mode="include"),
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
