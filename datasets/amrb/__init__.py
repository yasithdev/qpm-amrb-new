import json
import logging
import os
from typing import Literal, Tuple

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from ..transforms import AddGaussianNoise
from .amrb import AMRBDataset


# --------------------------------------------------------------------------------------------------------------------------------------------------
def create_data_loaders(
    version: int,
    label_type: Literal["class", "type", "strain", "gram"],
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    cv_k: int,
    cv_folds: int,
    cv_mode: Literal["k-fold", "leave-out"],
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:

    # define transforms
    transform = Compose(
        [
            ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.01),
        ]
    )

    # load AMRB data
    train_loader = DataLoader(
        dataset=AMRBDataset(
            data_root=data_root,
            version=version,
            # query params
            train=True,
            cv_k=cv_k,
            cv_folds=cv_folds,
            cv_mode=cv_mode,
            label_type=label_type,
            # projection params
            transform=transform,
        ),
        batch_size=batch_size_train,
    )

    test_loader = DataLoader(
        dataset=AMRBDataset(
            data_root=data_root,
            version=version,
            # query params
            train=False,
            cv_k=cv_k,
            cv_folds=cv_folds,
            cv_mode=cv_mode,
            label_type=label_type,
            # projection params
            transform=transform,
        ),
        batch_size=batch_size_test,
    )

    return train_loader, test_loader


# --------------------------------------------------------------------------------------------------------------------------------------------------
def get_info(
    version: Literal[1, 2],
    label_type: Literal["class", "type", "strain", "gram"],
    data_root: str,
    cv_mode: Literal["k-fold", "leave-out"],
    **kwargs,
) -> dict:

    src_info_path = os.path.join(data_root, f"AMRB{version}", "info.json")
    logging.info(f"Dataset info: {src_info_path}")
    with open(src_info_path, "r") as f:
        src_info: dict = json.load(f)["strains"]

    # ignore non-sequenced strains
    strain_labels = [x for x in src_info if src_info[x]["sequenced"] == True]

    # select the target labels
    if label_type == "strain":
        target_labels = strain_labels
    else:
        target_labels = [src_info[slabel][label_type] for slabel in strain_labels]

    num_labels = len(set(target_labels))

    if cv_mode == "leave-out":
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
