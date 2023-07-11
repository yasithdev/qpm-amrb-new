from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor

from .transforms import AddGaussianNoise, create_target_transform


def get_targets() -> List[str]:
    targets = sorted(map(str, range(10)))
    return targets


def get_shape() -> Tuple[int, int, int]:
    return (3, 32, 32)


def get_datasets(
    data_root: str,
    ood: List[int] = [],
) -> Tuple[Dataset, Dataset]:
    transform = Compose(
        [
            ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.01),
        ]
    )
    targets = get_targets()
    target_transform = create_target_transform(targets, ood)

    trainset = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    testset = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    return trainset, testset
