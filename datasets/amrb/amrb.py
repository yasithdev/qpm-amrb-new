import json
import logging
import os
from time import time
from typing import Callable, Dict, Literal, Optional, List, Tuple

import einops
import numpy as np
import torchvision


class AMRBDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_root: str,
        version: int,
        # query params
        train: bool,
        cv_k: int,
        cv_folds: int,
        cv_mode: Literal["k-fold", "leave-out"],
        label_type: Literal["class", "type", "strain", "gram"],
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
        ds_path = os.path.join(data_root, f"AMRB{version}")

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
            src_info: Dict[str, Dict[str, str]] = json.load(f)["strains"]

        # ignore non-sequenced strains
        strain_labels = [x for x in src_info if src_info[x]["sequenced"] == True]

        # ----------------------------------------------------------------- #
        # load genetic similarity                                           #
        # ----------------------------------------------------------------- #
        gensim_path = os.path.join(ds_path, "gensim_strain.csv")
        gensim = np.genfromtxt(gensim_path, delimiter=",", dtype=str)
        gensim_labels = [*map(str, gensim[0])]
        assert set(gensim_labels) == set(strain_labels)

        # sort gensim labels and values
        gensim_slabels = sorted(gensim_labels)
        perm = [*map(gensim_labels.index, gensim_slabels)]
        gensim_y = gensim[1:].astype(np.float32)[perm, :][:, perm]

        # ----------------------------------------------------------------- #
        # select gensim_labels for task (ignore the rest)                   #
        # ----------------------------------------------------------------- #
        slabels = gensim_slabels
        if label_type == "strain":
            tlabels = slabels
        else:
            tlabels = [src_info[x][label_type] for x in slabels]
        full_label_map = list(zip(slabels, tlabels))
        uniq_tlabels = sorted(set(tlabels))

        if cv_mode == "leave-out":
            assert cv_folds == len(uniq_tlabels)
            # select all tlabels where train  == (tlabel != kth label)
            cond: Callable[[str], bool] = lambda t: train == (t != uniq_tlabels[cv_k])
            self.labels = list(filter(lambda t: cond(t), uniq_tlabels))
            cv_label_map = list(filter(lambda s_t: cond(s_t[1]), full_label_map))
        else:
            # select all tlabels
            self.labels = uniq_tlabels
            cv_label_map = full_label_map
        logging.info(
            f"[preparation] selected labels for {cv_mode} crossval: {self.labels}"
        )

        # ----------------------------------------------------------------- #
        # generate data_x and data_y for sequential reading                 #
        # ----------------------------------------------------------------- #
        logging.info(f"Preparing dataset for reading")
        t_start = time()

        # ----------------------------------------------------------------- #
        # generate data dict                                                #
        # ----------------------------------------------------------------- #
        data_dict: Dict[str, np.ndarray] = {}

        for slabel, tlabel in cv_label_map:

            src_x = self.filter_outliers(src_data[slabel])

            if tlabel in data_dict:
                data_dict[tlabel] = np.concatenate(
                    (data_dict[tlabel], src_x),
                    axis=0,
                )
            else:
                data_dict[tlabel] = src_x
        logging.info("[preparation] gathered data and filtered outliers")

        # ----------------------------------------------------------------- #
        # if cv_mode is 'k-fold', select the k-th crossval fold       #
        # ----------------------------------------------------------------- #
        if cv_mode == "k-fold":
            # select the k-th subset of the data_dict
            for tlabel in data_dict:
                num_images = data_dict[tlabel].shape[0]
                shift = int(num_images * cv_k / cv_folds)
                pivot = int(num_images * split_frac)
                data_dict[tlabel] = np.roll(data_dict[tlabel], shift, axis=0)
                if train:
                    data_dict[tlabel] = data_dict[tlabel][:pivot]
                else:
                    data_dict[tlabel] = data_dict[tlabel][pivot:]
            logging.info(f"[preparation] selected the k-th crossval fold")

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

        data_x = np.stack([data_dict[label] for label in sorted(data_dict)], axis=1)
        data_x = einops.rearrange(data_x, "b l h w -> (b l) h w 1")
        logging.info("[preparation] generated data_x")

        N = len(uniq_tlabels)
        if cv_mode == "k-fold":
            data_y = np.tile(
                A=np.eye(N, dtype=np.float32),
                reps=(data_x.shape[0] // N, 1),
            )
        else:
            N -= 1
            if train:
                data_y = np.tile(
                    A=np.eye(N, dtype=np.float32),
                    reps=(data_x.shape[0] // N, 1),
                )
            else:
                # using the gensim of left-out labels to compute metrics
                gensim_vec = self.gensim(
                    full_label_map, uniq_tlabels, cv_k, gensim_y, N
                )
                logging.info(f"Gensim Vector: {gensim_vec}")
                data_y = np.tile(
                    A=gensim_vec,
                    reps=(data_x.shape[0], 1),
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

    def gensim(
        self,
        full_label_map: List[Tuple[str, str]],
        uniq_tlabels: List[str],
        cv_k: int,
        gensim_y: np.ndarray,
        N: int,
        mode: Literal[1, 2] = 1,
    ) -> np.ndarray:

        # split the two gensim measures into two symmetric matrices
        s1 = np.triu(gensim_y) + np.triu(gensim_y, 1).T
        s2 = np.tril(gensim_y) + np.tril(gensim_y, -1).T
        sim = (s1 + s2) / 2
        logging.info(f"Gensim Matrix: {sim.shape}")

        # compute the group indices
        group_idxs: Dict[str, List[int]] = {}
        for i, (_, tlabel) in enumerate(full_label_map):
            if tlabel in group_idxs:
                group_idxs[tlabel].append(i)
            else:
                group_idxs[tlabel] = [i]

        assert N == len(group_idxs) - 1, "mismatch in label counts!"

        # target probability vector
        pvec = np.zeros((1, N), dtype=np.float32)

        # generate similarity matrix
        tlabel_k = uniq_tlabels[cv_k]
        for tlabel in group_idxs:
            if tlabel != tlabel_k:
                k = uniq_tlabels.index(tlabel)
                if k > uniq_tlabels.index(tlabel_k):
                    k -= 1
                n = 0
                for i in group_idxs[tlabel_k]:
                    for j in group_idxs[tlabel]:
                        pvec[0, k] += sim[i, j]
                        n += 1
                pvec[0, k] /= n

        return pvec / pvec.sum()

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
