from . import amrb, cifar10, mnist, qpm

def get_data(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    ood: list[int],
):
    if dataset_name == "MNIST":
        dm = mnist.DataModule(data_dir, batch_size, ood)
    elif dataset_name == "CIFAR10":
        dm = cifar10.DataModule(data_dir, batch_size, ood)
    elif dataset_name.startswith("AMRB"):
        ver, label_type = dataset_name[4:].split("_", maxsplit=1)
        dm = amrb.DataModule(data_dir, batch_size, ood, ver, label_type)
    elif dataset_name.startswith("QPM"):
        label_type = dataset_name[4:]
        dm = qpm.DataModule(data_dir, batch_size, ood, label_type)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")

    return dm
