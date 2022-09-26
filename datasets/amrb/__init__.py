import json
import logging
import os
from typing import List, Literal, Tuple

import torchvision
from torch.utils.data import DataLoader

from ..transforms import AddGaussianNoise
from .amrb import AMRB


# --------------------------------------------------------------------------------------------------------------------------------------------------
def create_data_loaders(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    version: int,
    crossval_k: int,
    crossval_folds: int,
    crossval_mode: Literal["k-fold", "leave-out"],
    label_type: Literal["class", "type", "strain", "gram"],
) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), AddGaussianNoise(mean=0.0, std=0.01)]
    )

    # load AMRB data
    train_loader = DataLoader(
        dataset=AMRB(
            data_root=data_root,
            version=version,
            # query params
            train=True,
            crossval_k=crossval_k,
            crossval_folds=crossval_folds,
            crossval_mode=crossval_mode,
            label_type=label_type,
            # projection params
            transform=transform,
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=AMRB(
            data_root=data_root,
            version=version,
            # query params
            train=False,
            crossval_k=crossval_k,
            crossval_folds=crossval_folds,
            crossval_mode=crossval_mode,
            label_type=label_type,
            # projection params
            transform=transform,
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


# --------------------------------------------------------------------------------------------------------------------------------------------------
def get_info(
    data_root: str,
    version: Literal[1, 2],
    label_type: Literal["class", "type", "strain", "gram"],
    crossval_mode: Literal["k-fold", "leave-out"],
) -> dict:

    src_info_path = os.path.join(data_root, f"AMRB_{version}", "info.json")
    logging.info(f"Dataset info: {src_info_path}")
    with open(src_info_path, "r") as f:
        src_info: dict = json.load(f)["data"]

    class_labels = sorted(src_info)
    if label_type == "class":
        target_labels = class_labels
    else:
        target_labels = [src_info[clabel][label_type] for clabel in class_labels]

    num_labels = len(set(target_labels))

    if crossval_mode == "leave-out":
        return {
            "num_train_labels": num_labels - 1,
            "num_test_labels": 1,
        }
    else:
        return {
            "num_train_labels": num_labels,
            "num_test_labels": num_labels,
        }


# --------------------------------------------------------------------------------------------------------------------------------------------------
