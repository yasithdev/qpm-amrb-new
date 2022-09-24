import json
import logging
import os
from time import time
from typing import Callable, Dict, List, Literal, Optional

import einops
import numpy as np
import torchvision

CROSSVAL_MODE = 0
LEAVE_OUT_MODE = 1


class AMRB(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_root: str,
        version: int,
        # query params
        train: bool,
        crossval_k: int,
        crossval_folds: int,
        ood_labels: List[str],
        label_type: Literal["class", "type", "strain", "gram"],
        split_frac: float = 0.8,
        # projection params
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(data_root, None, transform, target_transform)
        assert version in [1, 2], "Unknown AMRB version: should be 1 or 2"

        if len(ood_labels) == 0:
            mode = CROSSVAL_MODE
        else:
            mode = LEAVE_OUT_MODE

        # ----------------------------------------------------------------- #
        # base path for dataset files                                       #
        # ----------------------------------------------------------------- #
        ds_path = os.path.join(data_root, f"AMRB_{version}")

        # ----------------------------------------------------------------- #
        # load dataset for sequential reading                               #
        # ----------------------------------------------------------------- #
        src_data_path = os.path.join(ds_path, "proc_data.npz")
        logging.info(f"Dataset file: {src_data_path}")
        src_data = np.load(src_data_path)

        # ----------------------------------------------------------------- #
        # load dataset info                                                 #
        # ----------------------------------------------------------------- #
        src_info_path = os.path.join(ds_path, "info.json")
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
        logging.info(f"Labels: {sorted(set(target_labels))}")

        zipped_labels = zip(class_labels, target_labels)
        if mode == CROSSVAL_MODE:
            # use all labels
            selected_labels = list(zipped_labels)
        else:
            # if training, use all labels except ood_labels
            if train == True:
                cond = lambda _, t: t not in ood_labels
            # if testing, only use ood_labels
            else:
                cond = lambda _, t: t in ood_labels
            selected_labels = list(filter(cond, zipped_labels))

        # ----------------------------------------------------------------- #
        # generate data_x and data_y for sequential reading                 #
        # ----------------------------------------------------------------- #
        logging.info(f"Preparing dataset for reading")
        t_start = time()

        # ----------------------------------------------------------------- #
        # generate x                                                        #
        # ----------------------------------------------------------------- #
        data_dict: Dict[str, np.ndarray] = {}

        for clabel, tlabel in selected_labels:

            source_x = self.filter_outliers(src_data[clabel])

            if tlabel in data_dict:
                data_dict[tlabel] = np.concatenate(
                    (data_dict[tlabel], source_x),
                    axis=0,
                )
            else:
                data_dict[tlabel] = source_x
        logging.info("[preparation] filtered outliers")

        # ----------------------------------------------------------------- #
        # if mode is cross-val, do a train/test split                       #
        # ----------------------------------------------------------------- #
        if mode == CROSSVAL_MODE:
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
            logging.info("[preparation] performed train/test split for cross-validation")

        # ----------------------------------------------------------------- #
        # balance the class distribution of x                               #
        # ----------------------------------------------------------------- #
        data_counts = [data_dict[x].shape[0] for x in data_dict]
        target_count = min(max(data_counts), min(data_counts) * 4)
        for tlabel in data_dict:
            data_dict[tlabel] = self.augment(
                source_images=data_dict[tlabel],
                target_count=target_count,
            )
        logging.info("[preparation] balanced class distribution")

        # ----------------------------------------------------------------- #
        # generate x and y                                                  #
        # ----------------------------------------------------------------- #
        self.labels = sorted(data_dict)

        data_x = np.stack([data_dict[label] for label in self.labels], axis=1)
        data_x = einops.rearrange(data_x, "b l h w -> (b l) h w 1")
        logging.info("[preparation] generated data_x")

        if mode == CROSSVAL_MODE:
            num_labels = len(self.labels)
            data_y = np.tile(
                A=np.eye(N=num_labels, dtype=np.float32),
                reps=(data_x.shape[0] // num_labels, 1),
            )
        else:
            # TODO the value of self.labels may be incorrect for vis.py
            num_labels = len(set(self.labels).difference(ood_labels))
            if train:
                data_y = np.tile(
                    A=np.eye(N=num_labels, dtype=np.float32),
                    reps=(data_x.shape[0] // num_labels, 1),
                )
            else:
                # assign equal probability to all classes (TODO improve this)
                data_y = np.full(
                    shape=(data_x.shape[0], num_labels),
                    fill_value=1 / num_labels,
                    dtype=np.float32,
                )
        logging.info("[preparation] generated data_y")

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
        logging.info(f"Prepared dataset in {t_end - t_start} s")

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

    def augment(
        self,
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
                np.arange(len(aug_pool)),
                size=target_count - source_count,
                replace=False,
            )
            aug_data = aug_pool[aug_idxs]

            # concatenate source images with aug_data
            target_images = np.concatenate([source_images, aug_data], axis=0)

        assert target_images.shape[0] == target_count
        return target_images

    def filter_outliers(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        x_out = x

        # filter 1 - remove samples with negative values
        x_out = x_out[np.min(x_out, axis=(1, 2)) >= 0]

        # filter 2 - drop samples with i_max > Q3 + 1.5 * IQR
        i_max = np.max(x_out, axis=(1, 2))
        q3 = np.percentile(i_max, 75)
        iqr = q3 - np.percentile(i_max, 25)
        x_out = x_out[i_max <= q3 + 1.5 * iqr]

        return x_out
