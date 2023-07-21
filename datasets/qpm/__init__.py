from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize

from ..transforms import create_target_transform, TileChannels2d
from .qpm import bacteria_dataset


def get_targets(
    target_label: str,
) -> List[str]:
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
    return targets


def get_shape() -> Tuple[int, int, int]:
    return (1, 40, 40)
    # return (3, 224, 224)


def get_datasets(
    target_label: str,
    data_root: str,
    ood: List[int],
) -> Tuple[Dataset, Dataset]:
    transform = Compose(
        [
            ToTensor(),
            # Resize((224, 224), antialias=True),  # type: ignore
            # TileChannels2d(3),
        ]
    )
    targets = get_targets(target_label=target_label)
    target_transform = create_target_transform(targets, ood, infer_index=False)

    trainset = bacteria_dataset(
        data_dir=data_root,
        type_="train",
        transform=transform,
        target_transform=target_transform,
        label_type=target_label,
        balance_data=False,
    )
    testset = bacteria_dataset(
        data_dir=data_root,
        type_="test",
        transform=transform,
        target_transform=target_transform,
        label_type=target_label,
        balance_data=False,
    )
    return trainset, testset
