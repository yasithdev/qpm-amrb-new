import json
import logging
import os
from time import time
from typing import Callable, Dict, List, Literal, Optional, Tuple

import einops
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from .transforms import AddGaussianNoise


class AMRB(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_root: str,
        version: int,
        # query params
        train: bool,
        crossval_k: int,
        crossval_folds: int,
        ood_labels: List[str] = [],
        label_type: Literal["class", "type", "strain", "gram"] = "class",
        split_frac: float = 0.8,
        # projection params
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(data_root, None, transform, target_transform)
        assert version in [1, 2], "Unknown AMRB version: should be 1 or 2"

        # ----------------------------------------------------------------- #
        # base path for dataset files                                       #
        # ----------------------------------------------------------------- #
        ds_path = os.path.join(data_root, f"AMRB_{version}")

        # ----------------------------------------------------------------- #
        # load dataset for sequential reading                               #
        # ----------------------------------------------------------------- #
        src_data_path = os.path.join(
            ds_path,
            "proc_data.npz",
        )
        logging.info(f"Dataset file: {src_data_path}")
        src_data = np.load(src_data_path)

        # ----------------------------------------------------------------- #
        # load dataset info                                                 #
        # ----------------------------------------------------------------- #
        src_info_path = os.path.join(
            ds_path,
            "info.json",
        )
        logging.info(f"Dataset info: {src_info_path}")
        with open(src_info_path, "r") as f:
            src_info: dict = json.load(f)["data"]

        # ----------------------------------------------------------------- #
        # select labels for task                                            #
        # ----------------------------------------------------------------- #
        class_labels = sorted(src_info)
        if label_type == "class":
            target_labels = class_labels
        else:
            target_labels = [src_info[x][label_type] for x in class_labels]

        if len(ood_labels) == 0:
            selected_labels = list(zip(class_labels, target_labels))
        else:
            selected_labels = []
            for clabel, tlabel in zip(class_labels, target_labels):
                if train == True and tlabel not in ood_labels:
                    selected_labels.append((clabel, tlabel))
                if train == False and tlabel in ood_labels:
                    selected_labels.append((clabel, tlabel))

        # ----------------------------------------------------------------- #
        # generate data_x and data_y for sequential reading                 #
        # ----------------------------------------------------------------- #
        logging.info(f"Optimizing dataset for reading")
        t_start = time()

        # ----------------------------------------------------------------- #
        # generate x                                                        #
        # ----------------------------------------------------------------- #
        data_dict: Dict[str, np.ndarray] = {}
        for clabel, tlabel in selected_labels:
            if tlabel in data_dict:
                data_dict[tlabel] = np.concatenate(
                    (data_dict[tlabel], src_data[clabel]),
                    axis=0,
                )
            else:
                data_dict[tlabel] = src_data[clabel]

        # ----------------------------------------------------------------- #
        # if no ood_labels given, do a train/test split                     #
        # ----------------------------------------------------------------- #
        if len(ood_labels) == 0:
            # roll images (for crossval) and train/test split
            for tlabel in data_dict:
                num_images = data_dict[tlabel].shape[0]
                shift = int(num_images * crossval_k / crossval_folds)
                pivot = int(num_images * split_frac)
                data_dict[tlabel] = np.roll(data_dict[tlabel], shift, axis=0)
                if train:
                    data_dict[tlabel] = data_dict[tlabel][:pivot]
                else:
                    data_dict[tlabel] = data_dict[tlabel][pivot:]

        # ----------------------------------------------------------------- #
        # balance the class distribution of x                               #
        # ----------------------------------------------------------------- #
        data_counts = [data_dict[x].shape[0] for x in data_dict]
        target_count = min(max(data_counts), min(data_counts) * 4)
        for tlabel in data_dict:
            data_dict[tlabel] = augment(
                source_images=data_dict[tlabel],
                target_count=target_count,
            )

        # ----------------------------------------------------------------- #
        # stack samples across labels                                       #
        # ----------------------------------------------------------------- #
        data_x = np.stack([data_dict[label] for label in sorted(data_dict)], axis=1)
        data_x = einops.rearrange(data_x, "b l h w -> (b l) h w 1")

        # ----------------------------------------------------------------- #
        # generate y                                                        #
        # ----------------------------------------------------------------- #
        if len(ood_labels) == 0:
            num_labels = len(set(target_labels))
            data_y = np.tile(
                A=np.eye(N=num_labels, dtype=np.float32),
                reps=(data_x.shape[0] // num_labels, 1),
            )
        else:
            num_labels = len(set(target_labels).difference(ood_labels))
            if train:
                data_y = np.tile(
                    A=np.eye(N=num_labels, dtype=np.float32),
                    reps=(data_x.shape[0] // num_labels, 1),
                )
            else:
                data_y = np.zeros(
                    shape=(data_x.shape[0], num_labels),
                    dtype=np.float32,
                )

        # ----------------------------------------------------------------- #
        # store generated dataset view for reading                          #
        # ----------------------------------------------------------------- #
        self.data_x = data_x
        self.data_y = data_y

        # ----------------------------------------------------------------- #
        # sanity check data                                                 #
        # ----------------------------------------------------------------- #
        assert self.data_x.shape[0] == self.data_y.shape[0]

        t_end = time()
        logging.info(f"Dataset optimized in {t_end - t_start} s")

    def __getitem__(self, index: int):

        img = self.data_x[index]
        target = self.data_y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:

        num_data = len(self.data_x)
        return num_data

    def get_info(self):
        raise NotImplementedError()


# --------------------------------------------------------------------------------------------------------------------------------------------------
def create_data_loaders(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    version: int,
    crossval_k: int,
    crossval_folds: int,
    ood_labels: List[str],
    label_type: Literal["class", "type", "strain", "gram"] = "class",
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
            ood_labels=ood_labels,
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
            ood_labels=ood_labels,
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
    crossval_k: int,
    crossval_folds: int,
    ood_labels: List[str],
    label_type: Literal["class", "type", "strain", "gram"],
    version: int,
) -> dict:
    if version == 1:
        num_labels = 7
    elif version == 2:
        num_labels = 21
    else:
        raise NotImplementedError(f"Unknown Dataset Version: {version}")

    if len(ood_labels) > 0:
        return {
            "num_train_labels": num_labels - len(ood_labels),
            "num_test_labels": len(ood_labels),
        }
    else:
        return {
            "num_train_labels": num_labels,
            "num_test_labels": num_labels,
        }


# --------------------------------------------------------------------------------------------------------------------------------------------------
def augment(
    source_images: np.ndarray,
    target_count: int,
) -> np.ndarray:

    source_count = source_images.shape[0]

    if source_count >= target_count:

        target_images = source_images[:target_count]

    else:

        # define a pool of augmented images (by rotating 90, 180, and 270)
        aug_pool = np.concatenate(
            [
                np.rot90(source_images, k=1, axes=(1, 2)),
                np.rot90(source_images, k=2, axes=(1, 2)),
                np.rot90(source_images, k=3, axes=(1, 2)),
            ],
            axis=0,
        )
        # randomly pick images from aug_pool (without replacement)
        aug_idxs = np.random.choice(
            np.arange(len(aug_pool)), size=target_count - source_count, replace=False
        )
        aug_data = aug_pool[aug_idxs]

        # concatenate source images with aug_data
        target_images = np.concatenate([source_images, aug_data], axis=0)

    assert target_images.shape[0] == target_count
    return target_images


# --------------------------------------------------------------------------------------------------------------------------------------------------
