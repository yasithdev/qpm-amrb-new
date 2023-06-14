from typing import Tuple

from torch.utils.data import DataLoader

from . import amrb, mnist, cifar10


def get_dataset_loaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == "MNIST":
        return mnist.create_data_loaders(**kwargs)
    elif dataset_name == "CIFAR10":
        return cifar10.create_data_loaders(**kwargs)
    elif dataset_name.startswith("AMRB"):
        version, label_type = dataset_name[4:].split('_', maxsplit=1)
        return amrb.create_data_loaders(int(version), label_type, **kwargs)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")


def get_dataset_chw(dataset_name: str) -> Tuple[int, int, int]:
    if dataset_name == "MNIST":
        return (1, 32, 32)
    elif dataset_name == "CIFAR10":
        return (3, 32, 32)
    elif dataset_name.startswith("AMRB"):
        return (1, 40, 40)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")


def get_dataset_info(dataset_name: str, **kwargs) -> dict:
    if dataset_name == "MNIST":
        return mnist.get_info(**kwargs)
    elif dataset_name == "CIFAR10":
        return cifar10.get_info(**kwargs)
    elif dataset_name.startswith("AMRB"):
        version, label_type = dataset_name[4:].split('_', maxsplit=1)
        return amrb.get_info(int(version), label_type, **kwargs)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")
