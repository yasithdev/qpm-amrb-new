import json
import logging
import os
from collections import defaultdict
from functools import partial
from itertools import accumulate
from time import time
from typing import Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


class InMemoryDataset(Dataset):
    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.data_x = data_x
        self.data_y = data_y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(
        self,
        index: int,
    ):
        img = self.data_x[index]
        target = self.data_y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(
        self,
    ) -> int:
        num_data = len(self.data_x)
        return num_data


def get_target_map(
    data_root: str,
    version: str,
    target_label: str,
) -> Dict[str, List[str]]:
    # data directory
    ds_path = os.path.join(data_root, f"AMRB{version}")
    # load metadata
    ds_meta_path = os.path.join(ds_path, "info.json")
    with open(ds_meta_path, "r") as f:
        strain_info: Dict[str, Dict[str, str]] = json.load(f)["strains"]
    # create target map
    target_map: Dict[str, List[str]] = defaultdict(lambda: [])
    for k, v in strain_info.items():
        v = k if target_label == "strain" else v[target_label]
        target_map[v].append(k)
    return target_map


def create_dataset(
    data_root: str,
    version: str,
    target_label: str,
    split_id: int,
    splits: list[float],
    balance_data: bool = False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    filter_labels: list[int] = [],
    filter_mode: str = "exclude",
) -> Dataset:
    # sanity checks
    assert sum(splits) == 1.0
    assert 0 <= split_id < len(splits)
    assert filter_mode in ["include", "exclude"]

    # start time
    t_start = time()

    # data directory
    ds_path = os.path.join(data_root, f"AMRB{version}")

    # load data
    ds_data_path = os.path.join(ds_path, "ctr_1_fit_f32.imag.npz")
    logging.info(f"Dataset file: {ds_data_path}")
    src_data: dict[str, np.ndarray] = np.load(ds_data_path)

    # load targets
    target_map = get_target_map(data_root, version, target_label)
    logging.info("[preparation] loaded target info")

    # some helper functions
    _concat = partial(np.concatenate, axis=0)

    # extract split
    sizes = [0] * len(target_map)
    tmp_data: dict[str, list[np.ndarray]] = defaultdict(lambda: [])

    for i, (target, strains) in enumerate(target_map.items()):
        # filtering criteria
        cond1 = filter_mode == "exclude"
        cond2 = i in filter_labels
        for strain in strains:
            strain_data = src_data[strain]
            B = strain_data.shape[0]
            bounds = [0] + [int(B * N) for N in accumulate(splits)]
            lo, hi = bounds[split_id], bounds[split_id + 1]
            split_data = strain_data[lo:hi]
            sizes[i] += B
            # filtering criteria
            if cond1 == cond2:
                continue
            tmp_data[target].append(split_data)
    logging.info("[preparation] extracted data split")

    data: dict[str, np.ndarray] = {}
    N = min(max(sizes), min(sizes) * 6)

    for k, v in tmp_data.items():
        v = _concat(v)
        if balance_data:
            data[k] = augment_by_rotation(v, N)
        else:
            data[k] = v

    # reference order of targets
    targets = sorted(target_map.keys())

    # prepare dataset
    X = _concat(list(data.values()))
    Y = _concat(list([targets.index(k)] * v.shape[0] for k, v in data.items()))
    dataset = InMemoryDataset(X, Y, transform, target_transform)

    # end time
    t_end = time()
    logging.info(f"Prepared dataset in {t_end - t_start} s")

    # return prepared datasets
    return dataset


def augment_by_rotation(
    source_images: np.ndarray,
    target_count: int,
) -> np.ndarray:
    # vars
    source_count = source_images.shape[0]
    diff = target_count - source_count

    if diff <= 0:
        target_images = source_images[: diff or None]

    else:
        # define a pool of augmented images
        aug_pool = np.concatenate(
            [
                np.rot90(source_images, k=1, axes=(1, 2)),  # 90 degree rotation
                np.rot90(source_images, k=2, axes=(1, 2)),  # 180 degree rotation
                np.rot90(source_images, k=3, axes=(1, 2)),  # 270 degree rotation
                source_images[:,::-1,:],                    # vertical flip
                source_images[:,:,::-1],                    # horizontal flip
            ],
            axis=0,
        )
        # randomly pick images from aug_pool (without replacement)
        aug_idxs = np.random.choice(len(aug_pool), size=diff, replace=False)
        aug_data = aug_pool[aug_idxs]

        # concatenate source images with aug_data
        target_images = np.concatenate([source_images, aug_data], axis=0)

    assert target_images.shape[0] == target_count
    return target_images
