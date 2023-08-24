from . import amrb, cifar10, mnist, qpm
import argparse

def get_data(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    ood: list[int],
    args : argparse.Namespace,
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
        dm = qpm.DataModule(data_dir, batch_size, ood, label_type, "supervised", args)
    elif dataset_name.startswith("S_QPM"): # e.g => S_QPM_strain
        label_type = dataset_name[6:]
        dm = qpm.DataModule(data_dir, batch_size, ood, label_type, "self_supervised", args)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is unsupported")

    return dm
