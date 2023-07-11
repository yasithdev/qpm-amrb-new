import json
import logging
import os
from collections import defaultdict
from functools import partial
from time import time
from typing import Callable, Dict, List, Optional, Tuple

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


def create_datasets(
    data_root: str,
    version: str,
    target_label: str,
    split_frac: float = 0.6,
    balance_train_data=False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset]:
    # start time
    t_start = time()

    # data directory
    ds_path = os.path.join(data_root, f"AMRB{version}")

    # load data
    ds_data_path = os.path.join(ds_path, "ctr_1_fit_f32.imag.npz")
    logging.info(f"Dataset file: {ds_data_path}")
    src_data = np.load(ds_data_path)

    # load targets
    target_map = get_target_map(data_root, version, target_label)
    logging.info("[preparation] loaded target info")

    # some helper functions
    _concat = partial(np.concatenate, axis=0)
    _bsize = lambda x: int(x.shape[0])

    # perform train/test split
    trn_data: Dict[str, np.ndarray] = {}
    tst_data: Dict[str, np.ndarray] = {}

    for target, strains in target_map.items():
        target_data = _concat([src_data[s] for s in strains])
        pivot = int(_bsize(target_data) * split_frac)
        trn_data[target], tst_data[target] = target_data[:pivot], target_data[pivot:]
    logging.info("[preparation] performed train/test split")
    # from here on, what's used are trn_data and tst_data

    if balance_train_data:
        # augment train data
        trn_sizes = [_bsize(d) for d in trn_data.values()]
        trn_N = min(max(trn_sizes), min(trn_sizes) * 4)
        trn_data = {k: augment_by_rotation(v, trn_N) for k, v in trn_data.items()}

    # reference order of targets
    targets = sorted(target_map.keys())

    # prepare train dataset
    trn_x = _concat(list(trn_data.values()))
    trn_y = _concat([[targets.index(k)] * _bsize(v) for k, v in trn_data.items()])
    trn_ds = InMemoryDataset(trn_x, trn_y, transform, target_transform)

    # prepare test dataset
    tst_x = _concat(list(tst_data.values()))
    tst_y = _concat([[targets.index(k)] * _bsize(v) for k, v in tst_data.items()])
    tst_ds = InMemoryDataset(tst_x, tst_y, transform, target_transform)

    # end time
    t_end = time()
    logging.info(f"Prepared datasets in {t_end - t_start} s")

    # return prepared datasets
    return trn_ds, tst_ds


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
        aug_idxs = np.random.choice(len(aug_pool), size=diff, replace=False)
        aug_data = aug_pool[aug_idxs]

        # concatenate source images with aug_data
        target_images = np.concatenate([source_images, aug_data], axis=0)

    assert target_images.shape[0] == target_count
    return target_images
