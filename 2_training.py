import logging
import os

from typing import Literal, Dict
import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

from config import Config, load_config, getenv
from models import get_model_optimizer_and_step
from models.common import load_saved_state, save_state, gen_epoch_acc

from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders


def gen_log_metrics(
    stats: Dict[str, np.ndarray],
    prefix: Literal["train", "test"],
    config: Config,
    step: int,
) -> dict:

    data = {}
    labels = config.labels
    mode = config.cv_mode
    assert labels

    y_true = stats["y_true"]
    y_pred = stats["y_pred"]
    B = y_pred.shape[0]
    K = len(labels)
    IMG = wandb.Image
    zero_ucty = np.zeros((B, 1), dtype=np.float32)
    y_ucty = stats.get("y_ucty", zero_ucty)
    samples = stats["samples"]

    # update y_pred and y_true for leave-out mode
    if mode == "leave-out":
        y_pred = np.concatenate([y_pred, y_ucty], axis=-1)
        if prefix == "test":
            y_true = np.full((B,), fill_value=K - 1)

    # log targets
    if mode == "leave-out" and prefix == "test":
        targets = [IMG(x, caption=labels[K - 1]) for x, _, _, _ in samples]
    else:
        targets = [IMG(x, caption=labels[y]) for x, y, _, _ in samples]
    data[f"{prefix}/targets"] = targets

    # log estimates
    estimates = [IMG(xz, caption=labels[yz]) for _, _, xz, yz in samples]
    data[f"{prefix}/estimates"] = estimates

    # log top-k acc and confusion matrix
    acc1, acc2, acc3 = gen_epoch_acc(y_pred, y_true, np.arange(K))  # type: ignore
    tqdm.write(
        f"[{prefix}] Epoch {step}: Loss(avg): {stats['loss']:.4f}, Acc: [{acc1:.4f}, {acc2:.4f}, {acc3:.4f}]"
    )
    data[f"{prefix}/acc1"] = acc1
    data[f"{prefix}/acc2"] = acc2
    data[f"{prefix}/acc3"] = acc3
    data[f"{prefix}/cm"] = wandb.plot.confusion_matrix(
        probs=y_pred,  # type: ignore
        y_true=y_true,  # type: ignore
        class_names=labels,
        title=f"Confusion Matrix ({prefix})",
    )
    data[f"{prefix}/roc"] = wandb.plot.roc_curve(
        y_true=y_true,
        y_probas=y_pred,
        labels=labels,
    )
    data[f"{prefix}/pr"] = wandb.plot.pr_curve(
        y_true=y_true,
        y_probas=y_pred,
        labels=labels,
    )
    done_keys = ["y_true", "y_pred", "samples"]
    for key in set(stats).difference(done_keys):
        val = stats[key]
        # unbounded histograms
        if key in ["u_norm", "v_norm", "z_norm", "z_nll"]:
            val = np.tanh(val)
            # save v_norm for AUROC computation
            if key in ["v_norm"]:
                data[f"{prefix}/{key}"] = val
            hist = np.histogram(val, bins=100, range=(0.0, 1.0))
            data[f"{prefix}/{key}_hist"] = wandb.Histogram(np_histogram=hist)
        # bounded histograms
        elif key == "y_ucty":
            hist = np.histogram(val, bins=100, range=(0.0, 1.0))
            data[f"{prefix}/{key}_hist"] = wandb.Histogram(np_histogram=hist)
        # log everything else
        else:
            data[f"{prefix}/{key}"] = val

    return data


def agg_log_metrics(
    log: dict,
) -> None:
    prefix = "ood_detection"
    if "train/v_norm" in log and "test/v_norm" in log:
        trn_v_norm = log.pop("train/v_norm")
        tst_v_norm = log.pop("test/v_norm")
        B_InD = trn_v_norm.shape[0]
        B_OoD = tst_v_norm.shape[0]
        # binary classification labels for ID and OOD
        LABELS = ["InD", "OoD"]
        values = np.concatenate([trn_v_norm, tst_v_norm], axis=0)
        values_2d = np.stack([1.0 - values, values], axis=-1)
        target = np.concatenate([np.zeros(B_InD), np.ones(B_OoD)], axis=0)
        log[f"{prefix}/roc"] = wandb.plot.roc_curve(target, values_2d, LABELS)
        log[f"{prefix}/pr"] = wandb.plot.pr_curve(target, values_2d, LABELS)
        log[f"{prefix}/auroc"] = metrics.roc_auc_score(target, values)


def main(config: Config):

    assert config.train_loader
    assert config.test_loader
    assert config.labels

    model, optim, step = get_model_optimizer_and_step(config)

    # load saved model and optimizer, if present
    load_saved_state(
        model=model,
        optim=optim,
        config=config,
    )
    model = model.float().to(config.device)

    wandb.watch(model, log_freq=100)

    # run train / test loops
    logging.info("Started Train/Test")

    artifact = wandb.Artifact(f"{config.run_name}-{config.model_name}", type="model")

    for epoch in range(config.train_epochs + 1):

        epoch_stats: dict = {}

        # training loop
        if epoch > 0:
            trn_stats = step(
                model=model,
                epoch=epoch,
                config=config,
                optim=optim,
            )
            log = gen_log_metrics(trn_stats, "train", config, epoch)
            epoch_stats.update(log)

        # testing loop
        tst_stats = step(
            model=model,
            epoch=epoch,
            config=config,
        )
        log = gen_log_metrics(tst_stats, "test", config, epoch)
        epoch_stats.update(log)

        agg_log_metrics(epoch_stats)

        wandb.log(epoch_stats, step=epoch)

        # save model and optimizer states
        save_state(
            model=model,
            optim=optim,
            config=config,
            epoch=epoch,
        )
        artifact.add_file(
            os.path.join(
                config.experiment_path, f"{config.model_name}_model_e{epoch}.pth"
            )
        )
        artifact.add_file(
            os.path.join(
                config.experiment_path, f"{config.model_name}_optim_e{epoch}.pth"
            )
        )

    artifact.save()


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)

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
    config.train_loader, config.test_loader = get_dataset_loaders(
        dataset_name=config.dataset_name,
        batch_size_train=config.batch_size,
        batch_size_test=config.batch_size,
        data_root=config.data_dir,
        cv_k=config.cv_k,
        cv_folds=config.cv_folds,
        cv_mode=config.cv_mode,
    )
    config.init_labels()

    import wandb
    import wandb.plot

    wandb.init(
        project="ood_capsule",
        name=config.run_name,
        config=config.run_config,
        notes = getenv("NOTES"),
    )

    main(config)
