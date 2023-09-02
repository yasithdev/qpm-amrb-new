import numpy as np
from torch.utils.data import Dataset


class NDArrayDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        targets: np.ndarray,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        self.values = values
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        # derive shape and labels
        self.shape = values.shape[-1]
        labels = np.unique(targets).tolist()
        if target_transform is not None:
            labels = sorted(set(map(target_transform, labels)))
        labels = list(map(str, labels))
        self.labels = labels

    def __getitem__(self, index):
        value = self.values[index]
        target = self.targets[index]

        if self.transform:
            value = self.transform(value)

        if self.target_transform:
            target = self.target_transform(target)
        return value, target

    def __len__(self):
        return self.targets.shape[0]
