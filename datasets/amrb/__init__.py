from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from ..transforms import AddGaussianNoise, create_target_transform
from .amrb import create_datasets, get_target_map


def get_targets(
    version: str,
    target_label: str,
    data_root: str,
) -> List[str]:
    target_map = get_target_map(data_root, version, target_label)
    targets = sorted(target_map.keys())
    return targets


def get_shape() -> Tuple[int, int, int]:
    return (1, 40, 40)


def get_datasets(
    version: str,
    target_label: str,
    data_root: str,
    ood: List[int] = [],
) -> Tuple[Dataset, Dataset]:
    transform = Compose(
        [
            ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.01),
        ]
    )
    targets = get_targets(
        version=version,
        target_label=target_label,
        data_root=data_root,
    )
    target_transform = create_target_transform(targets, ood, pre_indexed=True)

    trainset, testset = create_datasets(
        data_root=data_root,
        version=version,
        target_label=target_label,
        transform=transform,
        target_transform=target_transform,
    )

    return trainset, testset
