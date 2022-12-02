from typing import Tuple

from torch.utils.data import DataLoader

from . import amrb, mnist, cifar10


def get_dataset_loaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == "MNIST":
        return mnist.create_data_loaders(**kwargs)
    elif dataset_name == "CIFAR10":
        return cifar10.create_data_loaders(**kwargs)
    elif dataset_name == "AMRB_1":
        return amrb.create_data_loaders(version=1, **kwargs)
    elif dataset_name == "AMRB_2":
        return amrb.create_data_loaders(version=2, **kwargs)
    else:
        raise ValueError(
            f"Only MNIST, CIFAR10, AMRB_1, and AMRB_2 datasets are supported. (Given: {dataset_name})"
        )


def get_dataset_chw(dataset_name: str) -> Tuple[int, int, int]:
    if dataset_name == "MNIST":
        return (1, 32, 32)
    elif dataset_name == "CIFAR10":
        return (3, 32, 32)
    elif dataset_name == "AMRB_1":
        return (1, 40, 40)
    elif dataset_name == "AMRB_2":
        return (1, 40, 40)
    else:
        raise ValueError(
            f"Only MNIST, CIFAR10, AMRB_1, and AMRB_2 datasets are supported. (Given: {dataset_name})"
        )


def get_dataset_info(dataset_name: str, **kwargs) -> dict:
    if dataset_name == "MNIST":
        return mnist.get_info(**kwargs)
    elif dataset_name == "CIFAR10":
        return mnist.get_info(**kwargs)
    elif dataset_name == "AMRB_1":
        return amrb.get_info(version=1, **kwargs)
    elif dataset_name == "AMRB_2":
        return amrb.get_info(version=2, **kwargs)
    else:
        raise ValueError(
            f"Only MNIST, CIFAR10, AMRB_1, and AMRB_2 datasets are supported. (Given: {dataset_name})"
        )
