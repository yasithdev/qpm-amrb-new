from . import amrb, cifar10, embeddings, mnist, qpm


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
    labels = None
    if "QPM_species" in embedding_path:
        # map strains to species
        from .qpm import species_mapping, species

        target_transform = species_mapping.__getitem__
        labels = species

    return embeddings.DataModule(
        embedding_path=embedding_path,
        batch_size=batch_size,
        labels=labels,
        target_transform=target_transform,
    )
