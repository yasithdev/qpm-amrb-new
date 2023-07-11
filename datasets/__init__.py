from typing import Tuple, Optional

from torch.utils.data import DataLoader

from . import amrb, mnist, cifar10, qpm


def get_dataset_loaders(
    dataset_name: str,
    **kw,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    if dataset_name == "MNIST":
        return mnist.create_data_loaders(**kw)
    elif dataset_name == "CIFAR10":
        return cifar10.create_data_loaders(**kw)
    elif dataset_name.startswith("AMRB"):
        kw["version"], kw["target_label"] = dataset_name[4:].split("_", maxsplit=1)
        return amrb.create_data_loaders(**kw)
    elif dataset_name.startswith("QPM"):
        kw["target_label"] = dataset_name[4:]
        return qpm.create_data_loaders(**kw)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")


def get_dataset_chw(
    dataset_name: str,
    **kw,
) -> Tuple[int, int, int]:
    if dataset_name == "MNIST":
        return mnist.get_shape()
    elif dataset_name == "CIFAR10":
        return cifar10.get_shape()
    elif dataset_name.startswith("AMRB"):
        return amrb.get_shape()
    elif dataset_name.startswith("QPM"):
        return qpm.get_shape()
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")


def get_dataset_info(
    dataset_name: str,
    **kw,
) -> Tuple[set, set, list]:
    if dataset_name == "MNIST":
        return mnist.get_targets(**kw)
    elif dataset_name == "CIFAR10":
        return cifar10.get_targets(**kw)
    elif dataset_name.startswith("AMRB"):
        kw["version"], kw["target_label"] = dataset_name[4:].split("_", maxsplit=1)
        return amrb.get_targets(**kw)
    elif dataset_name.startswith("QPM"):
        kw["target_label"] = dataset_name[4:]
        return qpm.get_targets(**kw)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")
