from typing import Optional, Tuple

from torch.utils.data import DataLoader

from .dataloaders import get_bacteria_dataloaders


def get_targets(
    target_label: str,
    ood: list = [],
    **kwargs,
) -> Tuple[set, set, list]:
    strains = [
        "Acinetobacter",
        "B_subtilis",
        "E_coli_K12",
        "S_aureus",
        "E_coli_CCUG_17620",
        "E_coli_NCTC_13441",
        "E_coli_A2-39",
        "K_pneumoniae_A2-23",
        "S_aureus_CCUG_35600",
        "E_coli_101",
        "E_coli_102",
        "E_coli_104",
        "K_pneumoniae_210",
        "K_pneumoniae_211",
        "K_pneumoniae_212",
        "K_pneumoniae_240",
        "Acinetobacter_K12-21",
        "Acinetobacter_K48-42",
        "Acinetobacter_K55-13",
        "Acinetobacter_K57-06",
        "Acinetobacter_K71-71",
    ]
    species = [
        "Acinetobacter",
        "B subtilis",
        "E. coli",
        "K. pneumoniae",
        "S. aureus",
    ]
    if target_label == "species":
        targets = species
    elif target_label == "strain":
        targets = strains
    else:
        raise NotImplementedError()
    ood_targets = set([targets[i] for i in ood])
    ind_targets = set(targets).difference(ood_targets)
    targets = sorted(ind_targets) + sorted(ood_targets)
    return ind_targets, ood_targets, targets


def create_data_loaders(
    target_label: str,
    batch_size_train: int,
    batch_size_test: int,
    data_root: str,
    ood: list,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_loader, test_loader, ood_loader = get_bacteria_dataloaders(
        label_type=target_label,
        data_dir=data_root,
        train_batch_size=batch_size_train,
        test_batch_size=batch_size_test,
        expand_channels=False,
        one_hot=False,
        ood=ood,
    )
    return train_loader, test_loader, ood_loader


def get_shape():
    return (3, 224, 224)
