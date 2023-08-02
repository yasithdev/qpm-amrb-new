from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from .transforms import AddGaussianNoise, ZeroPad2D, create_target_transform


def get_targets() -> List[str]:
    targets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return targets


def get_shape() -> Tuple[int, int, int]:
    return (1, 32, 32)


def get_datasets(
    data_root: str,
    ood: List[int] = [],
) -> Tuple[Dataset, Dataset]:
    transform = Compose(
        [
            ToTensor(),
            ZeroPad2D(h1=2, h2=2, w1=2, w2=2),  # 28 x 28 -> 32 x 32
            AddGaussianNoise(mean=0.0, std=0.01),
        ]
    )
    targets = get_targets()
    target_transform = create_target_transform(targets, ood)

    trainset = MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    testset = MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    return trainset, testset
