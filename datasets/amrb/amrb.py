import json
import logging
import os
from functools import partial
from time import time
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np

from ..in_memory_dataset import InMemoryDataset


def get_targets(
    data_root: str,
    version: int,
    target_label: str,
    ood: list = [],
) -> Tuple[set, set, list, dict]:
    # data directory
    ds_path = os.path.join(data_root, f"AMRB{version}")
    # load metadata
    ds_meta_path = os.path.join(ds_path, "info.json")
    with open(ds_meta_path, "r") as f:
        strain_info: Dict[str, Dict[str, str]] = json.load(f)["strains"]
    # create target map
    target_map: Dict[str, List[str]] = {}
    for k, v in strain_info.items():
        v = k if target_label == "strain" else v[target_label]
        target_map.setdefault(v, []).append(k)
    targets = sorted(target_map.keys())
    ood_targets = set([targets[i] for i in ood])
    ind_targets = set(targets).difference(ood_targets)
    # order targets to ease classification
    targets = sorted(ind_targets) + sorted(ood_targets)
    return ind_targets, ood_targets, targets, target_map


def create_datasets(
    data_root: str,
    version: int,
    target_label: str,
    ood: list = [],
    split_frac: float = 0.6,
    balance_class_distribution=False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[InMemoryDataset, InMemoryDataset, Optional[InMemoryDataset]]:
    # start time
    t_start = time()

    # data directory
    ds_path = os.path.join(data_root, f"AMRB{version}")

    # load data
    ds_data_path = os.path.join(ds_path, "ctr_1_fit_f32.imag.npz")
    logging.info(f"Dataset file: {ds_data_path}")
    src_data = np.load(ds_data_path)

    # load targets
    ood_targets, targets, target_map = get_targets(
        data_root, version, target_label, ood
    )[1:]
    logging.info("[preparation] loaded target info")

    # some helper functions
    _concat = partial(np.concatenate, axis=0)
    size_b = lambda x: x.shape[0]
    idx_t = lambda x: targets.index(x)

    # generate ood and ind dict for downstream use
    ood_data = {}
    ind_data = {}
    for target, strains in target_map.items():
        if target in ood_targets:
            ood_data[target] = _concat([src_data[s] for s in strains])
        else:
            ind_data[target] = _concat([src_data[s] for s in strains])
    logging.info("[preparation] performed ind/ood split")

    # perform train/test split
    trn_data: Dict[str, np.ndarray] = {}
    tst_data: Dict[str, np.ndarray] = {}
    for target, data in ind_data.items():
        pivot = int(size_b(data) * split_frac)
        trn_data[target], tst_data[target] = data[:pivot], data[pivot:]
    logging.info("[preparation] performed train/test split")
    # from here on, what's used are trn_data, tst_data, and ood_data

    if balance_class_distribution:
        # augment ood data
        ood_sizes = [size_b(d) for d in ood_data.values()]
        ood_N = min(max(ood_sizes), min(ood_sizes) * 4) if len(ood_sizes) > 0 else 0
        ood_data = {k: augment_by_rotation(v, ood_N) for k, v in ood_data.items()}

        # augment train data
        trn_sizes = [size_b(d) for d in trn_data.values()]
        trn_N = min(max(trn_sizes), min(trn_sizes) * 4)
        trn_data = {k: augment_by_rotation(v, trn_N) for k, v in trn_data.items()}

        # augment test data
        tst_sizes = [size_b(d) for d in tst_data.values()]
        tst_N = min(max(tst_sizes), min(tst_sizes) * 4)
        tst_data = {k: augment_by_rotation(v, tst_N) for k, v in tst_data.items()}
        logging.info("[preparation] balanced class distributions")

    # prepare ood dataset
    ood_ds = None
    if len(ood_targets) > 0:
        ood_x = _concat(list(ood_data.values()))
        ood_y = _concat([np.full(size_b(v), idx_t(k)) for k, v in ood_data.items()])
        p = np.random.permutation(size_b(ood_x))
        ood_ds = InMemoryDataset(ood_x[p], ood_y[p], transform)

    # prepare train dataset
    trn_x = _concat(list(trn_data.values()))
    trn_y = _concat([np.full(size_b(v), idx_t(k)) for k, v in trn_data.items()])
    p = np.random.permutation(size_b(trn_x))
    trn_ds = InMemoryDataset(trn_x[p], trn_y[p], transform, target_transform)

    # prepare test dataset
    tst_x = _concat(list(tst_data.values()))
    tst_y = _concat([np.full(size_b(v), idx_t(k)) for k, v in tst_data.items()])
    p = np.random.permutation(size_b(tst_x))
    tst_ds = InMemoryDataset(tst_x[p], tst_y[p], transform, target_transform)

    # end time
    t_end = time()
    logging.info(f"Prepared datasets in {t_end - t_start} s")

    # return prepared datasets
    return trn_ds, tst_ds, ood_ds


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
