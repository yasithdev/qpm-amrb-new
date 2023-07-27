import logging
import os
from typing import Literal

import numpy as np
import torch
import sklearn.metrics
from tqdm import tqdm

from config import Config, load_config
from datasets import init_dataloaders, init_labels, init_shape
from models import get_model_optimizer_and_step
from models.common import gen_topk_accs, load_model_state, save_model_state


def update_epoch_stats(
    epoch_stats: dict,
    stats: dict,
    prefix: Literal["trn", "tid", "tod"],
) -> None:
    epoch_stats.update({f"{prefix}/{k}": v for k, v in stats.items()})


def epoch_stats_to_wandb(
    stats: dict,
    config: Config,
    step: int,
) -> dict:
    # initialize metric dict
    metrics = {}
    metrics["trn/loss"] = None
    metrics["tst/loss"] = [None, None]
    metrics["trn/acc"] = (None, None, None)
    metrics["tst/acc"] = (None, None, None)

    # get label information
    ind_labels, ood_labels = config.get_ind_labels(), config.get_ood_labels()
    Ki, Ko = len(ind_labels), len(ood_labels)

    figs = {}
    if "trn" in stats:
        metrics["trn/loss"] = stats["trn"]["loss"]
        y_true_trn: np.ndarray = stats["trn"]["y_true"]
        y_prob_trn: np.ndarray = stats["trn"]["y_prob"]
        metrics["trn/acc"] = gen_topk_accs(y_true_trn, y_prob_trn, Ki)
        figs["trn/cm"] = wandb.plot.confusion_matrix(
            y_true=y_true_trn,  # type: ignore
            probs=y_prob_trn,  # type: ignore
            class_names=ind_labels,
            title=f"Confusion Matrix (Train)",
        )
        figs["trn/roc"] = wandb.plot.roc_curve(
            y_true=y_true_trn,
            y_probas=y_prob_trn,
            labels=ind_labels,
        )
        figs["trn/prc"] = wandb.plot.pr_curve(
            y_true=y_true_trn,
            y_probas=y_prob_trn,
            labels=ind_labels,
        )
        if "y_ucty" in stats["trn"]:
            y_ucty_trn: np.ndarray = stats["trn"]["y_ucty"]

    y_true_tst = []
    y_prob_tst = []

    if "tid" in stats:
        y_true_tid: np.ndarray = stats["tid"]["y_true"]
        y_prob_tid: np.ndarray = stats["tid"]["y_prob"]
        if len(y_true_tid) > 0:
            metrics["tst/loss"][0] = stats["tid"]["loss"]
            metrics["tst/acc"] = gen_topk_accs(y_true_tid, y_prob_tid, Ki)
            y_true_tst.append(y_true_tid)
            y_prob_tst.append(y_prob_tid)
            if "y_ucty" in stats["tid"]:
                y_ucty_tid: np.ndarray = stats["tid"]["y_ucty"]

    if "tod" in stats:
        y_true_tod: np.ndarray = stats["tod"]["y_true"]
        y_prob_tod: np.ndarray = stats["tod"]["y_prob"]
        if len(y_true_tod) > 0:
            metrics["tst/loss"][1] = stats["tod"]["loss"]
            y_true_tst.append(y_true_tod)
            y_prob_tst.append(y_prob_tod)
            if "y_ucty" in stats["tod"]:
                y_ucty_tod: np.ndarray = stats["tod"]["y_ucty"]

    # concatenate y of tid and tod datasets
    y_true_tst = np.concatenate(y_true_tst, axis=0)
    y_prob_tst = np.concatenate(y_prob_tst, axis=0)

    # zero-pad y_prob_tst for ood targets
    y_prob_tst = np.pad(y_prob_tst, pad_width=((0, 0), (0, Ko)))
    perm_labels = ind_labels + ood_labels

    figs["tst/cm"] = wandb.plot.confusion_matrix(
        y_true=y_true_tst,  # type: ignore
        probs=y_prob_tst,  # type: ignore
        class_names=perm_labels,
        title=f"Confusion Matrix (Test)",
    )
    figs["tst/roc"] = wandb.plot.roc_curve(
        y_true=y_true_tst,
        y_probas=y_prob_tst,
        labels=perm_labels,
    )
    figs["tst/prc"] = wandb.plot.pr_curve(
        y_true=y_true_tst,
        y_probas=y_prob_tst,
        labels=perm_labels,
    )

    tqdm.write(f"Epoch {step}: {metrics}")

    join = lambda x: ",".join(map(str,x))
    fig_row_lbl = []
    fig_a_data = []
    fig_a_col_lbl = []
    fig_b_data = []
    fig_b_col_lbl = []

    if "trn" in stats:
        fig_row_lbl.append("trn")
        (E_x_trn, E_y_trn, E_xp_trn, E_yp_trn) = zip(*stats["trn"]["samples"])
        E_x_trn = np.concatenate(E_x_trn, axis=1)
        E_xp_trn = np.concatenate(E_xp_trn, axis=1)
        fig_a_data.append(E_x_trn)
        fig_a_col_lbl.append(join(E_y_trn))
        fig_b_data.append(E_xp_trn)
        fig_b_col_lbl.append(join(E_yp_trn))

    if "tid" in stats and len(stats["tid"]["y_true"]) > 0:
        fig_row_lbl.append("tid")
        (E_x_tid, E_y_tid, E_xp_tid, E_yp_tid) = zip(*stats["tid"]["samples"])
        E_x_tid = np.concatenate(E_x_tid, axis=1)
        E_xp_tid = np.concatenate(E_xp_tid, axis=1)
        fig_a_data.append(E_x_tid)
        fig_a_col_lbl.append(join(E_y_tid))
        fig_b_data.append(E_xp_tid)
        fig_b_col_lbl.append(join(E_yp_tid))

    if "tod" in stats and len(stats["tod"]["y_true"]) > 0:
        fig_row_lbl.append("tod")
        (E_x_tod, E_y_tod, E_xp_tod, E_yp_tod) = zip(*stats["tod"]["samples"])
        E_x_tod = np.concatenate(E_x_tod, axis=1)
        E_xp_tod = np.concatenate(E_xp_tod, axis=1)
        fig_a_data.append(E_x_tod)
        fig_a_col_lbl.append(join(E_y_tod))
        fig_b_data.append(E_xp_tod)
        fig_b_col_lbl.append(join(E_yp_tod))

    fig_a_data = np.concatenate(fig_a_data, axis=0)
    fig_b_data = np.concatenate(fig_b_data, axis=0)

    fig_row_lbl = ", ".join([f"R{i+1}={v}" for i, v in enumerate(fig_row_lbl)])
    fig_a_col_lbl = ", ".join([f"R{i+1}={v}" for i, v in enumerate(fig_a_col_lbl)])
    fig_b_col_lbl = ", ".join([f"R{i+1}={v}" for i, v in enumerate(fig_b_col_lbl)])
    fig_a_cap = f"sample inputs - rows: [{fig_row_lbl}] - targets: [{fig_a_col_lbl}]"
    fig_b_cap = f"sample output - rows: [{fig_row_lbl}] - targets: [{fig_b_col_lbl}]"

    figs["samples/input"] = wandb.Image(fig_a_data, caption=fig_a_cap)
    figs["samples/output"] = wandb.Image(fig_b_data, caption=fig_b_cap)

    done_keys = ["y_true", "y_prob", "y_ucty", "samples"]

    for prefix in ["trn", "tid", "tod"]:
        if prefix not in stats:
            continue
        prefix_stats = stats[prefix]
        for key in set(prefix_stats).difference(done_keys):
            val = prefix_stats[key]
            # unbounded histograms
            if key in ["u_norm", "v_norm", "z_norm", "z_nll"]:
                val = np.tanh(val)
                # save v_norm for AUROC computation
                if key in ["v_norm"]:
                    figs[f"{prefix}/{key}"] = val
                hist = np.histogram(val, bins=100, range=(0.0, 1.0))
                figs[f"{prefix}/{key}_hist"] = wandb.Histogram(np_histogram=hist)
            # bounded histograms
            elif key == "y_ucty":
                hist = np.histogram(val, bins=100, range=(0.0, 1.0))
                figs[f"{prefix}/{key}_hist"] = wandb.Histogram(np_histogram=hist)
            # log everything else
            else:
                figs[f"{prefix}/{key}"] = val

    prefix = "ood_detection"
    if "v_norm" in stats["tid"] and "v_norm" in stats["tod"]:
        tid_v_norm = stats["tid"]["v_norm"]
        tod_v_norm = stats["tod"]["v_norm"]
        B_InD = tid_v_norm.shape[0]
        B_OoD = tod_v_norm.shape[0]
        # binary classification labels for ID and OOD
        LABELS = ["InD", "OoD"]
        values = np.concatenate([tid_v_norm, tod_v_norm], axis=0)
        values_2d = np.stack([1.0 - values, values], axis=-1)
        target = np.concatenate([np.zeros(B_InD), np.ones(B_OoD)], axis=0)
        stats[f"{prefix}/roc"] = wandb.plot.roc_curve(target, values_2d, LABELS)
        stats[f"{prefix}/pr"] = wandb.plot.pr_curve(target, values_2d, LABELS)
        stats[f"{prefix}/auroc"] = sklearn.metrics.roc_auc_score(target, values)

    data = {}
    data.update(metrics)
    data.update(figs)
    return data


def main(config: Config):
    assert config.train_loader
    assert config.test_loader

    model, optim, step = get_model_optimizer_and_step(config)

    # load saved model and optimizer, if present
    load_model_state(model, config)
    model = model.float().to(config.device)

    wandb.watch(model, log_freq=100)

    # run train / test loops
    logging.info("Started Train/Test")

    artifact = wandb.Artifact(f"{config.run_name}-{config.model_name}", type="model")

    # loop over epochs
    for epoch in range(config.train_epochs + 1):
        epoch_stats: dict = {}

        # train
        if epoch > 0:
            trn_stats = step(
                prefix="train",
                model=model,
                epoch=epoch,
                config=config,
                optim=optim,
            )
            epoch_stats["trn"] = trn_stats

        # test (InD)
        tid_stats = step(
            prefix="test_ind",
            model=model,
            epoch=epoch,
            config=config,
        )
        epoch_stats["tid"] = tid_stats

        # test (OoD)
        tod_stats = step(
            prefix="test_ood",
            model=model,
            epoch=epoch,
            config=config,
        )
        epoch_stats["tod"] = tod_stats

        wandb.log(epoch_stats_to_wandb(epoch_stats, config, epoch), step=epoch)

        # save model and optimizer states
        save_model_state(model, config, epoch)
        fp = config.experiment_path
        model_name = config.model_name
        artifact.add_file(os.path.join(fp, f"{model_name}_model_e{epoch}.pth"))

    artifact.save()


if __name__ == "__main__":
    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)

    config = load_config()

    # initialize data attributes and loaders
    init_labels(config)
    init_shape(config)
    init_dataloaders(config)

    config.print_labels()

    import wandb.plot

    import wandb

    wandb.init(
        project="uq_ood",
        name=config.run_name,
        config=config.run_config,
    )

    main(config)
