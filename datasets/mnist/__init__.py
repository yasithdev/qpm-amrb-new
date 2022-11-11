from typing import Literal, Tuple

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from ..transforms import AddGaussianNoise, ZeroPad2D
from .mnist import MNISTDataset


def create_data_loaders(
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    cv_k: int,
    cv_folds: int,
    cv_mode: Literal["k-fold", "leave-out"],
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:

    # define transforms
    transform = Compose(
        [
            ToTensor(),
            ZeroPad2D(h1=2,h2=2,w1=2,w2=2),
            AddGaussianNoise(mean=0.0, std=0.01),
        ]
    )

    # load MNIST data
    train_loader = DataLoader(
        dataset=MNISTDataset(
            data_root=data_root,
            # query params
            train=True,
            cv_k=cv_k,
            cv_folds=cv_folds,
            cv_mode=cv_mode,
            # projection params
            transform=transform,
        ),
        batch_size=batch_size_train,
    )

    test_loader = DataLoader(
        dataset=MNISTDataset(
            data_root=data_root,
            # query params
            train=False,
            cv_k=cv_k,
            cv_folds=cv_folds,
            cv_mode=cv_mode,
            # projection params
            transform=transform,
        ),
        batch_size=batch_size_test,
    )

    return train_loader, test_loader


# --------------------------------------------------------------------------------------------------------------------------------------------------


def get_info(
    cv_mode: Literal["k-fold", "leave-out"],
    **kwargs,
) -> dict:

    num_labels = 10

    if cv_mode == "leave-out":
        return {
            "num_train_labels": num_labels - 1,
            "num_test_labels": 1,
        }
    else:
        return {
            "num_train_labels": num_labels,
            "num_test_labels": num_labels,
        }


# --------------------------------------------------------------------------------------------------------------------------------------------------
