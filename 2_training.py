import logging
import os
import shutil

from typing import Literal, Dict
import numpy as np
import torch
from tqdm import tqdm

from config import Config, load_config
from models import get_model_optimizer_and_step
from models.common import load_saved_state, save_state, gen_epoch_acc

from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders


def log_stats(
    stats: Dict[str, np.ndarray],
    labels: list,
    prefix: Literal["train", "test"],
    mode: str,
    step: int,
) -> dict:

    data = {}

    y_true = stats["y_true"]
    y_pred = stats["y_pred"]
    B = y_pred.shape[0]

    # uncertainty
    zero_ucty = np.zeros((B, 1), dtype=np.float32)
    y_ucty = stats.get("y_ucty", zero_ucty)
    x_ucty = stats.get("x_ucty", zero_ucty)

    # compute confusion matrix
    samples = stats["samples"]

    # in leave-out mode, add column for left-out class
    if mode == "leave-out":
        y_pred = np.concatenate([y_pred, y_ucty], axis=-1)

    estimates = [wandb.Image(xz, caption=labels[yz]) for _, _, xz, yz in samples]
    data[f"{prefix}_estimates"] = estimates

    if mode == "leave-out" and prefix == "test":
        K = len(labels)
        y_true = np.full((B,), fill_value=K - 1)
        targets = [wandb.Image(x, caption=labels[K - 1]) for x, _, _, _ in samples]
    else:
        targets = [wandb.Image(x, caption=labels[y]) for x, y, _, _ in samples]
    data[f"{prefix}_targets"] = targets

    # top-k accuracies
    acc1, acc2, acc3 = gen_epoch_acc(y_pred=y_pred.tolist(), y_true=y_true.tolist(), labels=labels)
    tqdm.write(
        f"[{prefix}] Epoch {step}: Loss(avg): {stats['loss']:.4f}, Acc: [{acc1:.4f}, {acc2:.4f}, {acc3:.4f}]"
    )
    data[f"{prefix}_acc1"] = acc1
    data[f"{prefix}_acc2"] = acc2
    data[f"{prefix}_acc3"] = acc3
    data[f"{prefix}_cm"] = wandb.plot.confusion_matrix(
        y_true=y_true,  # type: ignore
        probs=y_pred,  # type: ignore
        class_names=labels,
        title=f"Confusion Matrix ({prefix})",
    )

    # everything else
    for key in stats.keys():
        if key not in ["acc", "y_true", "y_pred", "samples"]:
            data[f"{prefix}_{key}"] = stats[key]

    return data


def main(config: Config):

    assert config.train_loader
    assert config.test_loader

    if not (config.exc_resume or config.exc_dry_run):
        shutil.rmtree(experiment_path, ignore_errors=True)
        os.makedirs(experiment_path, exist_ok=True)

    model, optim, step = get_model_optimizer_and_step(config)

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
    train_labels = list(config.train_loader.dataset.labels)  # type: ignore
    test_labels = list(config.test_loader.dataset.labels)  # type: ignore
    if config.cv_mode == "leave-out":
        print(f"Labels (train): {train_labels}")
        print(f"Labels (test): {test_labels}")
        assert set(train_labels).isdisjoint(test_labels)
        labels = [*train_labels, *test_labels]
    else:
        assert set(train_labels) == set(test_labels)
        labels = train_labels
        print(f"Labels (train, test): {train_labels}")

    for epoch in range(config.train_epochs + 1):

        # training loop
        if epoch > 0:
            stats = step(
                model=model,
                epoch=epoch,
                config=config,
                optim=optim,
            )
            wandb.log(
                log_stats(
                    stats, labels, prefix="train", mode=config.cv_mode, step=epoch
                ),
                step=epoch,
            )

        # testing loop
        stats = step(
            model=model,
            epoch=epoch,
            config=config,
        )
        wandb.log(
            log_stats(stats, labels, prefix="test", mode=config.cv_mode, step=epoch),
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
