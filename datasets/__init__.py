from typing import Tuple

from torch.utils.data import DataLoader

from . import amrb, mnist


def get_dataset_loaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == "MNIST":
        return mnist.create_data_loaders(**kwargs)
    elif dataset_name == "AMRB_1":
        return amrb.create_data_loaders(version=1, **kwargs)
    elif dataset_name == "AMRB_2":
        return amrb.create_data_loaders(version=2, **kwargs)
    else:
        raise ValueError(
            f"Only MNIST, AMRB_1, and AMRB_2 datasets are supported. (Given: {dataset_name})"
        )


def get_dataset_info(dataset_name: str, **kwargs) -> dict:
    if dataset_name == "MNIST":
        return mnist.get_info(**kwargs)
    elif dataset_name == "AMRB_1":
        return amrb.get_info(version=1, **kwargs)
    elif dataset_name == "AMRB_2":
        return amrb.get_info(version=2, **kwargs)
    else:
        raise ValueError(
            f"Only MNIST, AMRB_1, and AMRB_2 datasets are supported. (Given: {dataset_name})"
        )
