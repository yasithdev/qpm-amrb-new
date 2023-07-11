import logging
import os

import numpy as np
import torch

from config import Config, load_config
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
from vis import gen_umap, plot_samples


def main(config: Config):

    assert config.train_loader
    assert config.test_loader
    assert config.dataset_info is not None

    ind_targets, ood_targets, targets = config.dataset_info

    x_trn = []
    y_trn = []

    for img, target in config.train_loader:
        x_trn.append(img)
        y_trn.append(target)
        if len(y_trn) == 200:
            break

    x_tst = []
    y_tst = []

    for img, target in config.test_loader:
        x_tst.append(img)
        y_tst.append(target)
        if len(y_tst) == 200:
            break

    x_trn = np.concatenate(x_trn, axis=0)
    y_trn = np.concatenate(y_trn, axis=0)
    x_tst = np.concatenate(x_tst, axis=0)
    y_tst = np.concatenate(y_tst, axis=0)

    n_trn = len(y_trn)
    n_tst = len(y_tst)

    logging.info(f"training samples={n_trn} [{x_trn.shape}, {y_trn.shape}]")
    logging.info(f"testing samples={n_tst} [{x_tst.shape}, {y_tst.shape}]")

    # ----------------------------------
    # Preview - Samples
    # ----------------------------------

    logging.info("Plotting training samples")
    plot_samples(
        x=x_trn,
        y=y_trn,
        out_path=os.path.join(config.experiment_path, f"preview.trn.pdf"),
        labels=sorted(ind_targets),
    )

    logging.info("Plotting testing samples")
    plot_samples(
        x=x_tst,
        y=y_tst,
        out_path=os.path.join(config.experiment_path, f"preview.tst.pdf"),
        labels=sorted(ind_targets),
    )

    # ----------------------------------
    # UMAP - Samples
    # ----------------------------------
    logging.info("Plotting UMAP projection of training samples")
    gen_umap(
        x=x_trn,
        y=y_trn.astype(np.int32).argmax(-1),
        out_path=os.path.join(config.experiment_path, f"umap.trn.png"),
        title=f"UMAP: {n_trn} Training Samples",
        labels=sorted(ind_targets),
    )

    logging.info("Plotting UMAP projection of testing samples")
    gen_umap(
        x=x_tst,
        y=y_tst.astype(np.int32).argmax(-1),
        out_path=os.path.join(config.experiment_path, f"umap.tst.png"),
        title=f"UMAP: {n_tst} Testing Samples)",
        labels=sorted(ind_targets),
    )


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.random.manual_seed(42)

    config = load_config()

    # get dataset info
    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        data_root=config.data_dir,
        cv_mode=config.cv_mode,
    )
    # get image dims
    config.image_chw = get_dataset_chw(
        dataset_name=config.dataset_name,
    )
    # initialize data loaders
    config.train_loader, config.test_loader, config.ood_loader = get_dataset_loaders(
        dataset_name=config.dataset_name,
        batch_size_train=config.batch_size,
        batch_size_test=config.batch_size,
        data_root=config.data_dir,
        cv_k=config.cv_k,
        cv_folds=config.cv_folds,
        cv_mode=config.cv_mode,
    )
    config.print_labels()

    main(config)
