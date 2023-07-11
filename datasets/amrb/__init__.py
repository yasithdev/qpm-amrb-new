from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from ..transforms import AddGaussianNoise, TileChannels2d, ZeroPad2D
from .amrb import create_datasets
from .amrb import get_targets as __get_targets


def get_targets(
    target_label: str,
    version: str,
    data_root: str,
    ood: list = [],
    **kwargs,
) -> Tuple[set, set, list]:
    return __get_targets(data_root, int(version), target_label, ood)[:3]


def create_data_loaders(
    target_label: str,
    version: str,
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    ood: list = [],
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    # define transforms
    transform = Compose(
        [
            ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.01),
        ]
    )
    ind_targets, *_ = get_targets(target_label, version, data_root, ood)
    target_transform = Compose(
        [
            lambda x: torch.tensor(x),
            lambda x: F.one_hot(x, len(ind_targets)),
        ]
    )
    train_dataset, test_dataset, ood_dataset = create_datasets(
        data_root=data_root,
        version=int(version),
        target_label=target_label,
        ood=ood,
        transform=transform,
        target_transform=target_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test)
    ood_loader = None
    if ood_dataset is not None:
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size_test)

    return train_loader, test_loader, ood_loader


def get_shape():
    return (1, 40, 40)
