import os

import numpy as np
import torch

from config import Config, load_config
from util.vis import gen_umap, plot_samples


def main(config: Config):
    # ----------------------------------

    train_loader, test_loader = config.data_loader()

    experiment_name = "1_amrb_preview"
    experiment_path = os.path.join(
        config.experiment_dir,
        experiment_name,
        config.dataset_name,
        str(config.crossval_k),
    )
    vis_path = os.path.join(experiment_path, "vis")

    x_trn = []
    y_trn = []
    for img, target in train_loader:
        x_trn.append(img)
        y_trn.append(target)
        if len(y_trn) == 200:
            break

    x_tst = []
    y_tst = []
    for img, target in test_loader:
        x_tst.append(img)
        y_tst.append(target)
        if len(y_tst) == 200:
            break

    x_trn = np.concatenate(x_trn, axis=0)
    y_trn = np.concatenate(y_trn, axis=0)
    x_tst = np.concatenate(x_tst, axis=0)
    y_tst = np.concatenate(y_tst, axis=0)

    print(x_trn.shape, y_trn.shape)
    print(x_tst.shape, y_tst.shape)
    
    exit(0)

    # ----------------------------------
    # Preview - Samples
    # ----------------------------------
    print("Preview: Inputs")

    print("* TRN")
    plot_samples(x_trn, y_trn, os.path.join(vis_path, "preview.trn.pdf"))

    print("* VAL")
    plot_samples(x_val, y_val, os.path.join(vis_path, "preview.val.pdf"))

    print("* TST")
    plot_samples(x_tst, y_tst, os.path.join(vis_path, "preview.tst.pdf"))

    # ----------------------------------
    # UMAP - Samples
    # ----------------------------------
    print(f"Selecting 30k items from {DS_NAME}")
    (x_trn, y_trn, n_trn), (x_val, y_val, n_val), (x_tst, y_tst, n_tst) = take_n(
        data, limit=3e4
    )

    print("UMAP: Inputs")

    print("* TRN")
    gen_umap(
        x_trn,
        y_trn,
        f"UMAP - Training Samples ({n_trn})",
        os.path.join(vis_path, "umap.trn.png"),
    )

    print("* VAL")
    gen_umap(
        x_val,
        y_val,
        f"UMAP - Validation Samples ({n_trn})",
        os.path.join(vis_path, "umap.val.png"),
    )

    print("* TST")
    gen_umap(
        x_tst,
        y_tst,
        f"UMAP - Testing Samples ({n_trn})",
        os.path.join(vis_path, "umap.tst.png"),
    )


if __name__ == "__main__":
    # initialize the RNG deterministically
    np.random.seed(42)
    torch.random.manual_seed(42)
    config = load_config()
    main(config)
