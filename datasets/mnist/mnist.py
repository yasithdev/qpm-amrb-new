import random
from typing import Callable, Literal, Optional

import numpy as np
import torchvision


class MNISTDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_root: str,
        # query params
        train: bool,
        cv_k: int,
        cv_folds: int,
        cv_mode: Literal["k-fold", "leave-out"],
        split_frac: float = 0.6,
        # projection params
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # load original dataset
        src_train = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
        )
        src_test = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
        )
        # combine train and test data into dict form
        dataset_all = {}
        for dataset in [src_train, src_test]:
            for (img, target) in dataset:
                if target in dataset_all:
                    dataset_all[target].append(img)
                else:
                    dataset_all[target] = [img]
        targets = sorted(dataset_all)
        one_hot_targets = np.eye(len(targets), dtype=np.float32)
        # if set to k-fold crossval mode
        if cv_mode == "k-fold":
            dataset_sub = {}
            for target in targets:
                images = dataset_all[target]
                shift = int(len(images) * cv_k / cv_folds)
                pivot = int(len(images) * split_frac)
                images = images[shift:] + images[:shift]
                dataset_sub[target] = images[:pivot] if train else images[pivot:]
            self.labels = targets
        # if set to leave-out crossval mode
        elif cv_mode == "leave-out":
            assert cv_folds == len(targets)
            # drop cv_k'th target
            leave_out_target = targets.pop(cv_k)
            one_hot_targets = np.delete(one_hot_targets, cv_k, axis=1)
            if train:
                dataset_sub = {t: dataset_all[t] for t in targets}
                self.labels = targets
            else:
                dataset_sub = {leave_out_target: dataset_all[leave_out_target]}
                self.labels = [leave_out_target]
        else:
            raise NotImplementedError(f"Unsupported cv_mode: {cv_mode}")
        # create final dataset
        dataset_final = []
        for target in dataset_sub:
            for image in dataset_sub[target]:
                dataset_final.append((image, one_hot_targets[target].copy()))
        random.shuffle(dataset_final)
        # assign object variables
        self.dataset = dataset_final
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):

        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.dataset)
