from typing import Literal, Tuple

from torch.utils.data import DataLoader

from .dataloaders import get_bacteria_dataloaders

def create_data_loaders(
    label_type: str,
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    cv_k: int,
    cv_folds: int,
    cv_mode: str,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    train_loader, test_loader = get_bacteria_dataloaders(label_type=label_type, data_dir=data_root, train_batch_size=batch_size_train, expand_channels=False, one_hot=True)
    return train_loader, test_loader
    

def get_info(
    label_type: str,
    data_root: str,
    cv_mode: str,
    **kwargs,
) -> dict:
    if label_type == "species":
        num_labels = 5
    elif label_type == "strain":
        num_labels = 21
    else:
        raise ValueError(f"Unexpected label_type: {label_type}")
    
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