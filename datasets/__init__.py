from . import amrb, cifar10, embeddings, mnist, qpm, rbc, rot_mnist
from .qpm import species, species_mapping
from .rbc import classes, patient_to_binary_mapping
from .rot_mnist import rotation_labels, digit_to_rotation_mapping
from .mnist import labels as mnist_labels
from .cifar10 import labels as cifar10_labels
from .amrb import labels as amrb_labels
from .transforms import reindex_for_ood


def get_grouping(
    name: str,
    ood: list[int],
):
    if name.startswith("QPM_species"):
        labels, grouping = species, species_mapping
    elif name.startswith("RBC"):
        labels, grouping = classes, patient_to_binary_mapping
    elif name.startswith("rot_mnist"):
        labels, grouping = rotation_labels, digit_to_rotation_mapping
    elif name == "mnist":
        labels, grouping = mnist_labels, list(range(10)) # identity-grouping for now
    elif name == "cifar10":
        labels, grouping = cifar10_labels, list(range(10)) # identity-grouping for now
    else:
        raise ValueError()
    # ood adjustments (NOTE this will reindex the TARGET label)
    filter_targets = [i for i, x in enumerate(grouping) if x in ood]
    perm_labels = reindex_for_ood(labels, ood)
    perm_grouping = list(map(perm_labels.index, map(labels.__getitem__, grouping)))
    # sanity check
    if len(ood) == 0:
        assert perm_grouping == grouping
        assert perm_labels == labels
    return perm_grouping, perm_labels, filter_targets


def get_data(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    ood: list[int],
    aug_hw_224: bool = False,
    aug_ch_3: bool = False,
    apply_target_transform: bool = True, # NOTE set to false for permutation testing models
):
    perm_grouping, perm_labels, filter_targets = get_grouping(dataset_name, ood)

    args: dict = dict(
        data_root=data_dir,
        batch_size=batch_size,
        ood=filter_targets,
        target_transform=perm_grouping.__getitem__ if apply_target_transform else None,
    )

    if dataset_name == "MNIST":
        dm = mnist.DataModule(**args, aug_ch_3=aug_ch_3)
    elif dataset_name == "CIFAR10":
        dm = cifar10.DataModule(**args)
    elif dataset_name.startswith("QPM"):
        label_type = dataset_name[4:]
        dm = qpm.DataModule(**args, target_label=label_type, aug_hw_224=aug_hw_224, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("RBC"):
        image_type = dataset_name[4:]
        dm = rbc.DataModule(**args, target_label=image_type, aug_hw_224=True, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("rot_mnist"):
        dm = rot_mnist.DataModule(**args, aug_ch_3=aug_ch_3)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")

    return dm, perm_grouping, perm_labels


def get_embedding(
    emb_name: str,
    emb_dir: str,
    batch_size: int,
    ood: list[int],
    apply_target_transform: bool = True, # NOTE set to false for permutation testing models
):
    perm_grouping, perm_labels, filter_targets = get_grouping(emb_name, ood)

    dm = embeddings.DataModule(
        emb_dir=emb_dir,
        emb_name=emb_name,
        batch_size=batch_size,
        ood=filter_targets,
        target_transform=perm_grouping.__getitem__ if apply_target_transform else None,
    )
    return dm, perm_grouping, perm_labels
