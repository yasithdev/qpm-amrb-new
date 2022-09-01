import logging
import os
from time import time
from typing import Any, Callable, Optional, Tuple

import einops
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from .transforms import AddGaussianNoise


class AMRB(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool,
        version: int,
        crossval_k: int,
        ood_mode: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, None, transform, target_transform)

        self.train = train
        self.version = version
        self.crossval_k = crossval_k
        self.ood_mode = ood_mode

        assert version in [1, 2], "Unknown AMRB version: should be 1 or 2"

        # ----------------------------------------------------------------- #
        # if in leave-out mode or crossval train mode, use the test set.    #
        # if not, use the test set                                          #
        # ----------------------------------------------------------------- #
        ds = "train" if (self.ood_mode or self.train) else "test"
        self.x_path = os.path.join(
            self.root,
            f"AMRB_{self.version}",
            "proc_data",
            f"{ds}_k{self.crossval_k}.npz",
        )
        logging.info(f"Dataset file: {self.x_path}")

        self.src_data = np.load(self.x_path)
        self.labels = sorted(self.src_data)

        self.ood_label = "S_aureus_CCUG35600"
        self.ood_label_idx = self.labels.index(self.ood_label)

        self.num_labels = len(self.labels)
        self.num_data = sum([self.src_data[key].shape[0] for key in self.labels])

        # ----------------------------------------------------------------- #
        # generate data_x and data_y for sequential reading                 #
        # ----------------------------------------------------------------- #
        logging.info(f"Optimizing dataset for reading")
        t_start = time()

        # ----------------------------------------------------------------- #
        # case 1 - leave-out method                                         #
        # ----------------------------------------------------------------- #
        if self.ood_mode:

            # ----------------------------------------------------------------- #
            # define excluded labels                                            #
            # ----------------------------------------------------------------- #
            exc_labels = [self.ood_label]
            num_exc_labels = len(exc_labels)

            # ----------------------------------------------------------------- #
            # define included labels                                            #
            # ----------------------------------------------------------------- #
            inc_labels = [x for x in self.labels if x not in exc_labels]
            num_inc_labels = len(inc_labels)

            # ----------------------------------------------------------------- #
            # case 1.1 - training data                                          #
            # ----------------------------------------------------------------- #
            if self.train:

                # ----------------------------------------------------------------- #
                # generate x                                                        #
                # ----------------------------------------------------------------- #
                data_x = np.stack([self.src_data[key] for key in inc_labels], axis=1)

                # ----------------------------------------------------------------- #
                # generate y                                                        #
                # ----------------------------------------------------------------- #
                data_y = np.tile(
                    A=np.eye(N=num_inc_labels, dtype=np.float16),
                    reps=(data_x.shape[0] // num_inc_labels, 1),
                )

            # ----------------------------------------------------------------- #
            # case 1.2 - testing data                                           #
            # ----------------------------------------------------------------- #
            else:

                # ----------------------------------------------------------------- #
                # generate x                                                        #
                # ----------------------------------------------------------------- #
                data_x = np.stack([self.src_data[key] for key in exc_labels], axis=1)

                # ----------------------------------------------------------------- #
                # generate y                                                        #
                # ----------------------------------------------------------------- #
                # Here, data_y is an equi-probable tensor. For evaluation purposes, #
                # it would be better to use one of the methods below.               #
                # ----------------------------------------------------------------- #
                #   (a) probability encoding of the genetic similarity, or          #
                #   (b) one-hot encoding of the closest label.                      #
                # ----------------------------------------------------------------- #
                data_y = np.full(
                    shape=(len(data_x), num_inc_labels),
                    fill_value=1.0 / num_inc_labels,
                    dtype=np.float16,
                )

            # ----------------------------------------------------------------- #
            # rearrange x to have all classes in a batch                        #
            # ----------------------------------------------------------------- #
            data_x = einops.rearrange(data_x, "b l h w -> (b l) 1 h w")

        # ----------------------------------------------------------------- #
        # case 2 - cross validation method                                  #
        # ----------------------------------------------------------------- #
        else:

            # ----------------------------------------------------------------- #
            # generate x                                                        #
            # ----------------------------------------------------------------- #
            data_x = np.stack([self.src_data[key] for key in self.labels], axis=1)
            data_x = einops.rearrange(data_x, "b l h w -> (b l) 1 h w")

            # ----------------------------------------------------------------- #
            # generate y                                                        #
            # ----------------------------------------------------------------- #
            data_y = np.tile(
                A=np.eye(N=self.num_labels, dtype=np.float16),
                reps=(data_x.shape[0] // self.num_labels, 1),
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

    def __getitem__(self, index: int) -> Any:

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


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    version: int,
    ood_mode: bool,
    crossval_k: int,
) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), AddGaussianNoise(mean=0.0, std=0.01)]
    )

    # load AMRB data
    train_loader = DataLoader(
        dataset=AMRB(
            root=data_root,
            train=True,
            version=version,
            crossval_k=crossval_k,
            ood_mode=ood_mode,
            transform=transform,
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=AMRB(
            root=data_root,
            train=False,
            version=version,
            crossval_k=crossval_k,
            ood_mode=ood_mode,
            transform=transform,
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load_v1(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    ood_mode: bool,
    crossval_k: int,
) -> Tuple[DataLoader, DataLoader]:
    return load(
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        data_root=data_root,
        version=1,
        ood_mode=ood_mode,
        crossval_k=crossval_k,
    )


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load_v2(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    ood_mode: bool,
    crossval_k: int,
) -> Tuple[DataLoader, DataLoader]:
    return load(
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        data_root=data_root,
        version=2,
        ood_mode=ood_mode,
        crossval_k=crossval_k,
    )
