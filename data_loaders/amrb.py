import os
from typing import Tuple, Any, Optional, Callable

import numpy as np
import torch.utils.data
import torchvision

from data_loaders.util import AddGaussianNoise


class AMRB(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool,
        version: int,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        ood_mode: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.train = train
        self.version = version
        assert version in [1, 2], "Unknown AMRB version: should be 1 or 2"

        mode = "trn" if self.train else "tst"
        self.x_path = os.path.join(self.root, f"AMRB_{self.version}", f"{mode}_x.npy")
        self.y_path = os.path.join(self.root, f"AMRB_{self.version}", f"{mode}_y.npy")

        self.ood_mode = ood_mode
        if version == 1:
            self.ood_index = 1
            self.num_labels = 7
        elif version == 2:
            self.ood_index = 6
            self.num_labels = 21

        self.data_x = np.load(self.x_path)
        self.data_y = np.load(self.y_path)

        assert self.data_x.shape[0] == self.data_y.shape[0]

    def __getitem__(self, index: int) -> Any:

        if self.ood_mode:
            if self.train:
                block = index // (self.num_labels - 1)
                pos = index % (self.num_labels - 1)
                if pos >= self.ood_index:
                    pos += 1
                index = (block * self.num_labels) + pos
            else:
                index = (index * self.num_labels) + self.ood_index

        img = self.data_x[index]
        target = self.data_y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        num_data = self.data_y.shape[0]
        if self.ood_mode:
            if self.train:
                num_data = num_data // self.num_labels * (self.num_labels - 1)
            else:
                num_data = num_data // self.num_labels
        return num_data


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    version: int,
    ood_mode: bool,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), AddGaussianNoise(mean=0.0, std=0.01)]
    )

    # load AMRB data
    train_loader = torch.utils.data.DataLoader(
        dataset=AMRB(
            root=data_root,
            train=True,
            version=version,
            ood_mode=ood_mode,
            transform=transform,
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=AMRB(
            root=data_root,
            train=False,
            version=version,
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
    ood_mode: str,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return load(
        batch_size_train, batch_size_test, data_root, version=1, ood_mode=ood_mode
    )


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load_v2(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    ood_mode: str,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return load(
        batch_size_train, batch_size_test, data_root, version=2, ood_mode=ood_mode
    )


# --------------------------------------------------------------------------------------------------------------------------------------------------


# --------------- CODE FROM PREVIOUS IMPLEMENTATION -------------------

# # combine data into array
# print("generating x,y data for train and test sets")
# X_TRAIN = []
# Y_TRAIN = []
# X_TEST = []
# Y_TEST = []

# for i, target in enumerate(tqdm(targets, **tqdm_args)):
#   # create train dataset
#   x_trn, y_trn = aug_trn_data[target].astype(np.float32), np.empty(shape=(target_count_trn, 1), dtype=np.int32)
#   y_trn.fill(i)
#   X_TRAIN.append(x_trn)
#   Y_TRAIN.append(y_trn)
#   # create test dataset
#   x_tst, y_tst = aug_tst_data[target].astype(np.float32), np.empty(shape=(target_count_tst, 1), dtype=np.int32)
#   y_tst.fill(i)
#   X_TEST.append(x_tst)
#   Y_TEST.append(y_tst)

# print("rearranging + writing data to file")
# px = 'n l h w -> (n l) h w 1'
# py = 'n l 1 -> (n l)'
# jobs = [
#   (X_TRAIN, px, "x_train"),
#   (Y_TRAIN, py, "y_train"),
#   (X_TEST, px, "x_test"),
#   (Y_TEST, py, "y_test")
# ]
# split_dataset = {}
# for (d, p, key) in tqdm(jobs, **tqdm_args):
#   d = np.stack(d, axis=1)
#   d = einops.rearrange(d, p)
#   split_dataset[key] = d

# np.savez_compressed(os.path.join(experiment_path, f"dataset_k{k}.npz"), **split_dataset)

# ---------------------------------------------------------------------
