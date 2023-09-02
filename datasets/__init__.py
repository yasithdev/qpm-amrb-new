import numpy as np
from torch.utils.data import Dataset

from . import amrb, cifar10, embeddings, mnist, qpm


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


def get_data(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    ood: list[int],
    aug_hw_224: bool = False,
    aug_ch_3: bool = False,
):
    if dataset_name == "MNIST":
        dm = mnist.DataModule(data_dir, batch_size, ood, aug_ch_3=aug_ch_3)
    elif dataset_name == "CIFAR10":
        dm = cifar10.DataModule(data_dir, batch_size, ood)
    elif dataset_name.startswith("AMRB"):
        ver, label_type = dataset_name[4:].split("_", maxsplit=1)
        dm = amrb.DataModule(data_dir, batch_size, ood, ver, label_type, aug_hw_224=aug_hw_224, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("QPM"):
        label_type = dataset_name[4:]
        dm = qpm.DataModule(data_dir, batch_size, ood, label_type, aug_hw_224=aug_hw_224, aug_ch_3=aug_ch_3)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")

    return dm


def get_embedding(
    embedding_path: str,
    batch_size: int,
    target_transform=None,
):
    if "QPM_species" in embedding_path:
        # map strains to species
        from .qpm import species_mapping

        target_transform = species_mapping.__getitem__

    return embeddings.DataModule(
        embedding_path=embedding_path,
        batch_size=batch_size,
        target_transform=target_transform,
    )
