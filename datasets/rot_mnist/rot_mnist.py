import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class rot_mnist_dataset(Dataset):
    """
    A standard dataset class to get the rotated MNIST dataset.

    Args:
        data_dir   : data directory which contains data hierarchy
        type_      : whether dataloader is train/val/test
        transform  : torchvision.transforms
        target_transform  : torchvision.transforms
        label_type : whether the mnist digit is rotated or not
    """

    def __init__(
        self,
        data_dir: str,
        type_: str,
        transform=None,
        target_transform=None,
        filter_labels=[],
        filter_mode: str = "exclude",
    ):
        # validation
        assert type_ in ["train", "val", "test"]
        assert filter_mode in ["include", "exclude"]

        # params
        self.transform = transform
        self.target_transform = target_transform
        self.filter_labels = filter_labels
        self.filter_mode = filter_mode
        self.type_ = type_
        print(f"Dataset type {type_}")

        # files : train_data.npz, test_data.npz, val_data.npz
        data = glob.glob(f"{data_dir}/{self.type_}_data.npz")

        # sanity check
        assert data['images'].shape[0] == data['images'].shape[1] == 28
        assert data['labels'].shape[0] == 1
        
        self.images = []
        
        transposed_images = np.transpose(data['images'], (3, 0, 1, 2)) #bring the batch dimension to the front
        transposed_labels = np.transpose(data['labels'], (1, 0))       #bring the batch dimension to the front
         
        self.images.extend(transposed_images)
        self.targets = [int(x) for x in transposed_labels.ravel()]

        assert len(self.images) == len(self.targets)
        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __must_filter(self, i) -> bool:
        cond1 = self.filter_mode == "exclude"
        cond2 = i in self.filter_labels
        return cond1 == cond2

    def __getitem__(self, idx):
        image, orig = self.images[idx], self.targets[idx]

        if self.transform:
            image     = self.transform(image)
        
        target = orig
        if self.target_transform:
            target = self.target_transform(target)

        return image, target, orig