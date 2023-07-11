from typing import Optional, Callable
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
