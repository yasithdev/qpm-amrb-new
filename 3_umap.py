import os
from typing import Literal

import numpy as np
import pandas as pd
import torch

from config import Config, load_config
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
from models import get_model_optimizer_and_step
from models.common import load_saved_state
from vis import gen_umap


def gen_log_metrics(
    stats: dict,
    prefix: Literal["train", "test"],
    config: Config,
    epoch: int,
) -> None:

    labels = config.labels
    mode = config.cv_mode
    assert labels

    y_true = np.array(stats["y_true"])
    y_pred = np.array(stats["y_pred"])
    z_pred = np.array(stats["z_pred"])
    B = y_pred.shape[0]
    K = len(labels)
    zero_ucty = np.zeros((B, 1), dtype=np.float32)
    y_ucty = stats.get("y_ucty", zero_ucty)
    x_ucty = stats.get("x_ucty", zero_ucty)

    # update y_pred and y_true for leave-out mode
    if mode == "leave-out":
        y_pred = np.concatenate([y_pred, y_ucty], axis=-1)
        if prefix == "test":
            y_true = np.full((B,), fill_value=K - 1)

    # log UMAP plot
    plot_title = f"UMAP (Z) - Test Set"
    plot_path = os.path.join(config.experiment_path, f"umapz_e{epoch}.png")
    gen_umap(
        x=z_pred,
        y=y_true.astype(np.int32),
        out_path=plot_path,
        title=plot_title,
        labels=labels,
    )


def main(
    config: Config,
) -> None:

    assert config.train_loader
    assert config.test_loader
    assert config.labels

    model, optim, step = get_model_optimizer_and_step(config)

    df_path = f"results/{config.run_name}_cls_summary.csv"
    df = pd.read_csv(df_path)

    rows = df[(df.model == config.model_name) & (df.cv_k == config.cv_k)]
    epochs = list(rows.step)
    print(f"Epochs: {epochs}")

    # for each selected epoch, load saved parameters
    for epoch in epochs:

        print(f"[UMAP] Epoch {epoch}...")

        load_saved_state(
            model=model,
            optim=optim,
            config=config,
            epoch=epoch,
        )
        model = model.float().to(config.device)

        # testing loop
        test_stats = step(
            model=model,
            epoch=epoch,
            config=config,
        )

        gen_log_metrics(test_stats, "test", config, epoch)


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)

    config = load_config()

    # get dataset info
    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        cv_mode=config.cv_mode,
        data_root=config.data_dir,
    )
    # get image dims
    config.image_chw = get_dataset_chw(
        dataset_name=config.dataset_name,
    )
    # initialize data loaders
    config.train_loader, config.test_loader = get_dataset_loaders(
        dataset_name=config.dataset_name,
        cv_mode=config.cv_mode,
        data_root=config.data_dir,
        batch_size_train=config.batch_size,
        batch_size_test=config.batch_size,
        cv_k=config.cv_k,
        cv_folds=config.cv_folds,
    )
    config.init_labels()

    main(config)
