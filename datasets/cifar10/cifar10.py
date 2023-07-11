from typing import Callable, Optional, Tuple

import numpy as np
import torchvision

from ..in_memory_dataset import InMemoryDataset


def get_targets(
    ood: list = [],
) -> Tuple[set, set, list]:
    targets = list(range(10))
    ood_targets = set(ood)
    ind_targets = set(targets).difference(ood_targets)
    targets = sorted(ind_targets) + sorted(ood_targets)
    return ind_targets, ood_targets, targets


def create_datasets(
    data_root: str,
    ood: list = [],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[InMemoryDataset, InMemoryDataset, Optional[InMemoryDataset]]:
    ood_targets = get_targets(ood)[1]

    ood_x = []
    ood_y = []

    ds_train = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
    )
    train_x = []
    train_y = []
    for x, y in ds_train:
        if y in ood_targets:
            ood_x.append(x)
            ood_y.append(y)
        else:
            train_x.append(x)
            train_y.append(y)

    test_x = []
    test_y = []
    ds_test = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
    )
    for x, y in ds_test:
        if y in ood_targets:
            ood_x.append(x)
            ood_y.append(y)
        else:
            test_x.append(x)
            test_y.append(y)

    ood_dataset = None
    if len(ood_x) > 0:
        ood_x = np.stack(ood_x, axis=0)
        ood_y = np.stack(ood_y, axis=0)
        ood_dataset = InMemoryDataset(ood_x, ood_y, transform)
    
    train_x = np.stack(train_x, axis=0)
    train_y = np.stack(train_y, axis=0)
    train_dataset = InMemoryDataset(train_x, train_y, transform, target_transform)
    
    test_x = np.stack(test_x, axis=0)
    test_y = np.stack(test_y, axis=0)
    test_dataset = InMemoryDataset(test_x, test_y, transform, target_transform)

    return train_dataset, test_dataset, ood_dataset
