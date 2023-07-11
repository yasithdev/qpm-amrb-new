from torch.utils.data import ChainDataset, DataLoader, Dataset, Subset

from config import Config

from . import amrb, cifar10, mnist, qpm


def init_labels(config: Config) -> None:
    if config.dataset_name == "MNIST":
        config.labels = mnist.get_targets()
    elif config.dataset_name == "CIFAR10":
        config.labels = cifar10.get_targets()
    elif config.dataset_name.startswith("AMRB"):
        version, target_label = config.dataset_name[4:].split("_", maxsplit=1)
        config.labels = amrb.get_targets(version, target_label, config.data_dir)
    elif config.dataset_name.startswith("QPM"):
        target_label = config.dataset_name[4:]
        config.labels = qpm.get_targets(target_label)
    else:
        raise ValueError(f"Dataset '{config.dataset_name}' is unsupported")


def init_shape(config: Config) -> None:
    if config.dataset_name == "MNIST":
        config.image_chw = mnist.get_shape()
    elif config.dataset_name == "CIFAR10":
        config.image_chw = cifar10.get_shape()
    elif config.dataset_name.startswith("AMRB"):
        config.image_chw = amrb.get_shape()
    elif config.dataset_name.startswith("QPM"):
        config.image_chw = qpm.get_shape()
    else:
        raise ValueError(f"Dataset '{config.dataset_name}' is unsupported")


def init_dataloaders(config: Config) -> None:
    if config.dataset_name == "MNIST":
        trainset, testset = mnist.get_datasets(config.data_dir, config.ood)
    elif config.dataset_name == "CIFAR10":
        trainset, testset = cifar10.get_datasets(config.data_dir, config.ood)
    elif config.dataset_name.startswith("AMRB"):
        version, target_label = config.dataset_name[4:].split("_", maxsplit=1)
        trainset, testset = amrb.get_datasets(
            version, target_label, config.data_dir, config.ood
        )
    elif config.dataset_name.startswith("QPM"):
        target_label = config.dataset_name[4:]
        trainset, testset = qpm.get_datasets(target_label, config.data_dir, config.ood)
    else:
        raise ValueError(f"Dataset '{config.dataset_name}' is unsupported")

    # create data loaders from train and test sets
    dataloaders = __make_dataloaders(trainset, testset, config)
    (config.train_loader, config.test_loader, config.ood_loader) = dataloaders


def __make_dataloaders(
    trainset: Dataset,
    testset: Dataset,
    config: Config,
):
    print("Performing ind/ood split")
    train_ood_idx = []
    train_ind_idx = []
    for idx, (_, target) in enumerate(trainset):  # type: ignore
        if target in config.ood:
            train_ood_idx.append(idx)
        else:
            train_ind_idx.append(idx)
    test_ood_idx = []
    test_ind_idx = []
    for idx, (_, target) in enumerate(testset):  # type: ignore
        if target in config.ood:
            test_ood_idx.append(idx)
        else:
            test_ind_idx.append(idx)
    print("Performed ind/ood split")

    train_oodset = Subset(trainset, train_ood_idx)
    trainset = Subset(trainset, train_ind_idx)
    test_oodset = Subset(testset, test_ood_idx)
    testset = Subset(testset, test_ind_idx)
    oodset = ChainDataset([train_oodset, test_oodset])

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=config.batch_size)
    ood_loader = DataLoader(oodset, batch_size=config.batch_size)

    return train_loader, test_loader, ood_loader
