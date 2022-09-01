import functools
from typing import Callable, Tuple

from torch.utils.data import DataLoader

from .amrb import load_v1 as load_amrb_1
from .amrb import load_v2 as load_amrb_2
from .mnist import load as load_mnist

DataLoaderFunction = Callable[[], Tuple[DataLoader, DataLoader]]


def get_data_loader(dataset_name: str, **kwargs) -> DataLoaderFunction:
    if dataset_name == "MNIST":
        return functools.partial(load_mnist, **kwargs)
    elif dataset_name == "AMRB_1":
        return functools.partial(load_amrb_1, **kwargs)
    elif dataset_name == "AMRB_2":
        return functools.partial(load_amrb_2, **kwargs)
    else:
        raise ValueError(
            f"Only MNIST, AMRB_1, and AMRB_2 datasets are supported. (Given: {dataset_name})"
        )
