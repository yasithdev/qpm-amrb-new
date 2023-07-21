import logging
import os

import numpy as np
import torch

from config import Config, load_config
from datasets import init_labels, init_shape, init_dataloaders
from vis import gen_umap, plot_samples


def main(config: Config):
    assert config.train_loader
    assert config.test_loader

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
        labels=config.get_ind_labels(),
    )

    logging.info("Plotting testing samples")
    plot_samples(
        x=x_tst,
        y=y_tst,
        out_path=os.path.join(config.experiment_path, f"preview.tst.pdf"),
        labels=config.get_ind_labels(),
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
        labels=config.get_ind_labels(),
    )

    logging.info("Plotting UMAP projection of testing samples")
    gen_umap(
        x=x_tst,
        y=y_tst.astype(np.int32).argmax(-1),
        out_path=os.path.join(config.experiment_path, f"umap.tst.png"),
        title=f"UMAP: {n_tst} Testing Samples)",
        labels=config.get_ind_labels(),
    )


if __name__ == "__main__":
    # initialize the RNG deterministically
    np.random.seed(42)
    torch.random.manual_seed(42)

    config = load_config()

    # initialize data attributes and loaders
    init_labels(config)
    init_shape(config)
    init_dataloaders(config)

    config.print_labels()

    main(config)
