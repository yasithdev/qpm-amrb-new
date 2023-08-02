from config import Config

from . import amrb, cifar10, mnist, qpm


def get_data(config: Config) -> None:
    if config.dataset_name == "MNIST":
        dm = mnist.DataModule(config.data_dir, config.batch_size, config.ood)
    elif config.dataset_name == "CIFAR10":
        dm = cifar10.DataModule(config.data_dir, config.batch_size, config.ood)
    elif config.dataset_name.startswith("AMRB"):
        ver, label = config.dataset_name[4:].split("_", maxsplit=1)
        dm = amrb.DataModule(config.data_dir, config.batch_size, config.ood, ver, label)

    elif config.dataset_name.startswith("QPM"):
        label = config.dataset_name[4:]
        dm = qpm.DataModule(config.data_dir, config.batch_size, config.ood, label)
    else:
        raise ValueError(f"Dataset '{config.dataset_name}' is unsupported")

    config.datamodule = dm
    config.labels = dm.targets
    config.image_chw = dm.shape
