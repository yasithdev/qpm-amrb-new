from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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
    ind_labels = config.get_ind_labels()
    ood_labels = config.get_ood_labels()
    Ki, Ko = len(ind_labels), len(ood_labels)
    permuted_idx = ind_labels + ood_labels
    bsize = config.batch_size

    print("Performing ind/ood split")
    trn_od_idx = []
    trn_id_idx = []
    for idx, (_, target) in enumerate(trainset):  # type: ignore
        if permuted_idx[target] in ood_labels:
            trn_od_idx.append(idx)
        else:
            trn_id_idx.append(idx)
    tst_od_idx = []
    tst_id_idx = []
    for idx, (_, target) in enumerate(testset):  # type: ignore
        if permuted_idx[target] in ood_labels:
            tst_od_idx.append(idx)
        else:
            tst_id_idx.append(idx)
    print("Performed ind/ood split")

    trn = Subset(trainset, trn_id_idx)
    tst = Subset(testset, tst_id_idx)
    ood = ConcatDataset([Subset(trainset, trn_od_idx), Subset(testset, tst_od_idx)])

    trn_loader = DataLoader(trn, bsize, shuffle=True, drop_last=True, num_workers=4)
    tst_loader = DataLoader(tst, bsize, shuffle=True, drop_last=True, num_workers=4)
    ood_loader = DataLoader(ood, bsize, shuffle=Ko > 0, drop_last=True, num_workers=4)

    print(len(trn_loader) * bsize, len(tst_loader) * bsize, len(ood_loader) * bsize)
    return trn_loader, tst_loader, ood_loader
