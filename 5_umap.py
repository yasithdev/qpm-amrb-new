import os

import numpy as np
import pandas as pd
import torch

from typing import Literal

from config import Config, load_config
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
from models import get_model_optimizer_and_step
from models.common import load_saved_state
from vis import gen_umap

def log_stats(
    stats: dict,
    labels: list,
    prefix: Literal["train", "test"],
    mode: str,
    plot_title: str,
    plot_path: str,
) -> None:

    loss = stats["loss"]
    acc1, acc2, acc3 = stats["acc"]

    y_true = np.array(stats["y_true"])
    y_pred = np.array(stats["y_pred"])
    z_pred = np.array(stats["z_pred"])

    # in leave-out mode, add column for left-out class
    if mode == "leave-out":
        y_pred = np.pad(y_pred, ((0, 0), (0, 1)), constant_values=0)

    if mode == "leave-out" and prefix == "test":
        n = len(labels) - 1
        y_true = np.full(y_true.shape[0], n)

    gen_umap(
        x=z_pred,
        y=y_true,
        out_path=plot_path,
        title=plot_title,
        labels=labels,
    )


def main(
    df: pd.DataFrame,
    config: Config,
) -> None:
    
    assert config.train_loader
    assert config.test_loader

    model, optim, step = get_model_optimizer_and_step(config)

    rows = df[
        (df.model == config.model_name) & (df.cv_k == config.cv_k)
    ]
    epochs = list(rows.step)
    print(f"Epochs: {epochs}")

    train_labels = list(config.train_loader.dataset.labels) # type: ignore
    test_labels = list(config.test_loader.dataset.labels) # type: ignore
    labels = (
        [*train_labels, *test_labels] if config.cv_mode == "leave-out" else train_labels
    )

    # for each selected epoch, load saved parameters
    for epoch in epochs:

        print(f"[UMAP] Epoch {epoch}...")

        load_saved_state(
            model=model,
            optim=optim,
            experiment_path=weights_path,
            config=config,
            epoch=epoch,
        )
        model = model.float().to(config.device)

        # testing loop
        stats = step(
            model=model,
            epoch=epoch,
            config=config,
        )
        
        plot_id = f"{config.dataset_name}-{config.label_type}-{config.cv_mode}-{config.model_name}-CV{config.cv_k}-E{epoch}"
        plot_path = os.path.join(
            config.experiment_dir,
            "5_umap",
            f"{plot_id}.png",
        )

        log_stats(
            stats,
            labels,
            prefix="test",
            mode=config.cv_mode,
            plot_title=f"[UMAP] Z: {plot_id} (test)",
            plot_path=plot_path,
        )


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(mode=True)

    config = load_config()

    # get dataset info
    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        label_type=config.label_type,
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
        label_type=config.label_type,
        cv_mode=config.cv_mode,
        data_root=config.data_dir,
        batch_size_train=config.batch_size,
        batch_size_test=config.batch_size,
        cv_k=config.cv_k,
        cv_folds=config.cv_folds,
    )

    weights_path = os.path.join(
        config.experiment_dir,
        "2_training",
        f"{config.dataset_name}-{config.model_name}",
        f"{config.label_type}-{config.cv_k}",
    )

    df_path = f"results/{config.dataset_name}_cls_summary_{config.cv_mode}_{config.label_type}.csv"
    df = pd.read_csv(df_path)

    main(df, config)
