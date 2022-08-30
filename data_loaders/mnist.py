from typing import Tuple

import torch.utils.data
import torchvision

from .util import AddGaussianNoise

# --------------------------------------------------------------------------------------------------------------------------------------------------


def load(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    ood_mode: bool,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), AddGaussianNoise(mean=0.0, std=0.01)]
    )

    # load MNIST data
    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


# --------------------------------------------------------------------------------------------------------------------------------------------------
