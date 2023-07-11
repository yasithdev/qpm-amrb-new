from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from ..transforms import AddGaussianNoise
from .cifar10 import create_datasets
from .cifar10 import get_targets as __get_targets


def get_targets(
    ood: list = [],
    **kwargs,
) -> Tuple[set, set, list]:
    return __get_targets(ood)


def create_data_loaders(
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
    ind_targets, ood_targets, targets = get_targets(ood)
    target_transform = Compose(
        [
            lambda x: targets.index(x),
            lambda x: torch.tensor(x),
            lambda x: F.one_hot(x, len(ind_targets)),
        ]
    )
    train_dataset, test_dataset, ood_dataset = create_datasets(
        data_root=data_root,
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
    return (3, 32, 32)
