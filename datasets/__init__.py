from . import cifar10, embeddings, mnist, qpm, rbc, rot_mnist
from .cifar10 import labels as cifar10_labels
from .mnist import labels as mnist_labels
from .qpm import species, species_alt, species_mapping, species_mapping_alt
from .rbc import classes, patient_to_binary_mapping
from .rot_mnist import digit_to_rotation_mapping, rotation_labels
from .transforms import reindex_for_ood


def get_grouping(
    name: str,
    ood: list[int],
):
    if name.startswith("QPM_species"):
        labels, grouping = species, species_mapping
    elif name.startswith("QPM2_species"):
        labels, grouping = species_alt, species_mapping_alt
    elif name.startswith("RBC") or name.startswith("rbc"):
        labels, grouping = classes, patient_to_binary_mapping
    elif "rot_mnist" in name:
        labels, grouping = rotation_labels, digit_to_rotation_mapping
    elif name == "MNIST":
        labels, grouping = mnist_labels, list(range(10)) # identity-grouping for now
    elif name == "CIFAR10":
        labels, grouping = cifar10_labels, list(range(10)) # identity-grouping for now
    else:
        raise ValueError()
    # ood adjustments (NOTE this will reindex the TARGET label)
    filter_targets = [i for i, x in enumerate(grouping) if x in ood]
    perm_labels = reindex_for_ood(labels, ood)
    lget = lambda x: labels[x] if x is not None else None
    iget = lambda l: perm_labels.index(l) if l is not None else None
    perm_grouping = list(map(iget, map(lget, grouping)))
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
    shuffle_training_data: bool = True,
):
    perm_grouping, perm_labels, filter_targets = get_grouping(dataset_name, ood)

    args: dict = dict(
        data_root=data_dir,
        batch_size=batch_size,
        ood=filter_targets,
        target_transform=perm_grouping.__getitem__ if apply_target_transform else None,
        shuffle_training_data=shuffle_training_data,
    )

    if dataset_name == "MNIST":
        dm = mnist.DataModule(**args, aug_ch_3=aug_ch_3)
    elif dataset_name == "CIFAR10":
        dm = cifar10.DataModule(**args)
    elif dataset_name.startswith("QPM_"):
        label_type = dataset_name[4:]
        dm = qpm.DataModule(**args, target_label=label_type, aug_hw_224=aug_hw_224, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("QPM2_"):
        label_type = dataset_name[5:]
        dm = qpm.DataModule(**args, target_label=label_type, aug_hw_224=aug_hw_224, aug_ch_3=aug_ch_3, strainwise_split=True)
    elif dataset_name.startswith("RBC_"):
        image_type = dataset_name[4:]
        dm = rbc.DataModule(**args, target_label=image_type, aug_hw_224=True, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("RBC2_"):
        image_type = dataset_name[5:]
        dm = rbc.DataModule(**args, target_label=image_type, aug_hw_224=True, aug_ch_3=aug_ch_3, patientwise_split=True)
    elif dataset_name.startswith("rbc_"):
        image_type = dataset_name[4:]
        dm = rbc.DataModule(**args, target_label=image_type, aug_hw_224=False, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("rbc2_"):
        image_type = dataset_name[5:]
        dm = rbc.DataModule(**args, target_label=image_type, aug_hw_224=False, aug_ch_3=aug_ch_3, patientwise_split=True)
    elif dataset_name.startswith("rot_mnist"):
        dm = rot_mnist.DataModule(**args, aug_ch_3=aug_ch_3)
    elif dataset_name.startswith("fake_rot_mnist"):
        dm = rot_mnist.DataModule(**args, aug_ch_3=aug_ch_3, fake_mode=True)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")

    return dm, perm_grouping, perm_labels


def get_embedding(
    emb_name: str,
    emb_dir: str,
    batch_size: int,
    ood: list[int],
    apply_target_transform: bool = True, # NOTE set to false for permutation testing models
    shuffle_training_data: bool = True,
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
