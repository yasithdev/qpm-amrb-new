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
    emb_name: str,
    emb_dir: str,
    batch_size: int,
    ood: list[int],
):
    if "QPM_species" in emb_name:
        # map strains to species
        from .qpm import species, species_mapping, strains

        source_labels = strains
        grouping = species_mapping
        labels = species
    else:
        raise ValueError()

    return embeddings.DataModule(
        emb_dir=emb_dir,
        emb_name=emb_name,
        batch_size=batch_size,
        source_labels=source_labels,
        grouping=grouping,
        labels=labels,
        ood=ood,
    )
