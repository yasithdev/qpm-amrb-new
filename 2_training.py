import logging
import os
import shutil

from typing import Literal
import numpy as np
import torch

from config import Config, load_config
from models import get_model_optimizer_and_loops
from models.common import load_saved_state, save_state

from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders


def log_stats(
    stats: dict,
    labels: list,
    prefix: Literal["train", "test"],
    mode: str,
) -> dict:
    
    data = {}

    # loss
    data[f"{prefix}_loss"] = stats["loss"]

    # top-k accuracies
    acc1, acc2, acc3 = stats["acc"]
    data[f"{prefix}_accuracy"] = acc1
    data[f"{prefix}_top2accuracy"] = acc2
    data[f"{prefix}_top3accuracy"] = acc3

    # optional parameters
    if "u_pred" in stats:
        data[f"{prefix}_u_pred"] = np.array(stats["u_pred"])
    if "v_pred" in stats:
        data[f"{prefix}_v_pred"] = np.array(stats["v_pred"])
    if "z_pred" in stats:
        data[f"{prefix}_z_pred"] = np.array(stats["z_pred"])
    if "z_nll" in stats:
        data[f"{prefix}_z_nll"] = np.array(stats["z_nll"])
    if "y_nll" in stats:
        data[f"{prefix}_y_nll"] = np.array(stats["y_nll"])

    # compute confusion matrix
    y_true = np.array(stats["y_true"])
    y_pred = np.array(stats["y_pred"])
    samples = stats["samples"]

    # in leave-out mode, add column for left-out class
    if mode == "leave-out":
        y_pred = np.pad(y_pred, ((0, 0), (0, 1)), constant_values=0)

    estimates = [wandb.Image(xz, caption=labels[yz]) for _, _, xz, yz in samples]
    data[f"{prefix}_estimates"] = estimates

    if mode == "leave-out" and prefix == "test":
        n = len(labels) - 1
        y_true = np.full(y_true.shape[0], n)
        targets = [wandb.Image(x, caption=labels[n]) for x, _, _, _ in samples]
    else:
        targets = [wandb.Image(x, caption=labels[y]) for x, y, _, _ in samples]
    data[f"{prefix}_targets"] = targets

    data[f"{prefix}_cm"] = wandb.plot.confusion_matrix(
        y_true=y_true,  # type: ignore
        probs=y_pred,  # type: ignore
        class_names=labels,
        title=f"Confusion Matrix ({prefix})",
    )

    return data


def main(config: Config):

    if not (config.exc_resume or config.exc_dry_run):
        shutil.rmtree(experiment_path, ignore_errors=True)
        os.makedirs(experiment_path, exist_ok=True)

    model, optim, train_model, test_model = get_model_optimizer_and_loops(config)

    # load saved model and optimizer, if present
    if config.exc_resume:
        load_saved_state(
            model=model,
            optim=optim,
            experiment_path=experiment_path,
            config=config,
        )
    model = model.float().to(config.device)

    wandb.watch(model, log_freq=100)

    # run train / test loops
    logging.info("Started Train/Test")
    train_labels = list(config.train_loader.dataset.labels)
    test_labels = list(config.test_loader.dataset.labels)
    labels = (
        [*train_labels, *test_labels] if config.cv_mode == "leave-out" else train_labels
    )

    for epoch in range(config.train_epochs + 1):

        # training loop
        if epoch > 0:
            stats = train_model(
                model=model,
                epoch=epoch,
                config=config,
                optim=optim,
            )
            wandb.log(
                log_stats(stats, labels, prefix="train", mode=config.cv_mode),
                step=epoch,
            )

        # testing loop
        stats = test_model(
            model=model,
            epoch=epoch,
            config=config,
        )
        wandb.log(
            log_stats(stats, labels, prefix="test", mode=config.cv_mode),
            step=epoch,
        )

        # save model/optimizer states
        if not config.exc_dry_run:
            save_state(
                model=model,
                optim=optim,
                experiment_path=experiment_path,
                epoch=epoch,
            )

        artifact.add_file(os.path.join(experiment_path, f"model_e{epoch}.pth"))
        artifact.add_file(os.path.join(experiment_path, f"optim_e{epoch}.pth"))


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(mode=True)

    config = load_config()

    # get dataset info
    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        data_root=config.data_dir,
        cv_mode=config.cv_mode,
        label_type=config.label_type,
    )
    # get image dims
    config.image_chw = get_dataset_chw(
        dataset_name=config.dataset_name,
    )
    # initialize data loaders
    config.train_loader, config.test_loader = get_dataset_loaders(
        dataset_name=config.dataset_name,
        batch_size_train=config.batch_size,
        batch_size_test=config.batch_size,
        data_root=config.data_dir,
        cv_k=config.cv_k,
        cv_folds=config.cv_folds,
        cv_mode=config.cv_mode,
        label_type=config.label_type,
    )

    # set experiment name and path
    experiment_name = "2_training"
    experiment_path = os.path.join(
        config.experiment_dir,
        experiment_name,
        f"{config.dataset_name}-{config.model_name}",
        f"{config.label_type}-{config.cv_k}",
    )
    name_tags = [
        config.dataset_name,
        config.model_name,
        config.cv_mode,
        str(config.cv_k),
    ]
    run_config = {
        "cv_folds": config.cv_folds,
        "cv_k": config.cv_k,
        "cv_mode": config.cv_mode,
        "dataset": config.dataset_name,
        "model": config.model_name,
    }
    if config.dataset_name.startswith("AMRB"):
        name_tags.insert(1, config.label_type)
        run_config["label_type"] = config.label_type
    run_name = "-".join(name_tags)

    import wandb
    import wandb.plot

    wandb.init(
        project="qpm-amrb-v2",
        name=run_name,
        config=run_config,
    )
    artifact = wandb.Artifact(run_name, type="model")
    main(config)
    artifact.save()
