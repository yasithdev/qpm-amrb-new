import logging
import os

import numpy as np
import torch
import sklearn.metrics
from tqdm import tqdm

from config import Config, load_config
from datasets import init_dataloaders, init_labels, init_shape
from models import get_model_optimizer_and_step
from models.common import gen_topk_accs, load_model_state, save_model_state
from matplotlib import pyplot as plt
from PIL import Image


def epoch_stats_to_wandb(
    stats: dict[str, dict],
    config: Config,
    step: int,
) -> dict:
    # initialize metric dict
    metrics = {}
    metrics["trn/loss"] = None
    metrics["trn/acc"] = dict(acc1=None, acc2=None, acc3=None)
    metrics["tid/loss"] = None
    metrics["tid/acc"] = dict(acc1=None, acc2=None, acc3=None)
    metrics["tod/loss"] = None

    # get label information
    ind_labels, ood_labels = config.get_ind_labels(), config.get_ood_labels()
    Ki, Ko = len(ind_labels), len(ood_labels)

    figs = {}
    if "trn" in stats:
        metrics["trn/loss"] = stats["trn"]["loss"]
        y_true_trn: np.ndarray = stats["trn"]["y_true"]
        y_prob_trn: np.ndarray = stats["trn"]["y_prob"]
        metrics["trn/acc"] = gen_topk_accs(y_true_trn, y_prob_trn, Ki)

        cm = sklearn.metrics.confusion_matrix(
            y_true_trn, y_prob_trn.argmax(-1), labels=list(range(Ki))
        )
        disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=ind_labels)
        disp.plot()
        fp = os.path.join(config.experiment_path, f"cm_trn_e{step}.png")
        plt.savefig(fp)
        plt.close()
        figs["trn/cm"] = wandb.Image(Image.open(fp))

    y_true_tst = []
    y_prob_tst = []

    if "tid" in stats:
        y_true_tid: np.ndarray = stats["tid"]["y_true"]
        y_prob_tid: np.ndarray = stats["tid"]["y_prob"]
        if len(y_true_tid) > 0:
            metrics["tid/loss"] = stats["tid"]["loss"]
            metrics["tid/acc"] = gen_topk_accs(y_true_tid, y_prob_tid, Ki)
            y_true_tst.append(y_true_tid)
            y_prob_tst.append(y_prob_tid)

    if "tod" in stats:
        y_true_tod: np.ndarray = stats["tod"]["y_true"]
        y_prob_tod: np.ndarray = stats["tod"]["y_prob"]
        if len(y_true_tod) > 0:
            metrics["tod/loss"] = stats["tod"]["loss"]
            y_true_tst.append(y_true_tod)
            y_prob_tst.append(y_prob_tod)

    # concatenate y of tid and tod datasets
    y_true_tst = np.concatenate(y_true_tst, axis=0)
    y_prob_tst = np.concatenate(y_prob_tst, axis=0)

    # zero-pad y_prob_tst for ood targets
    y_prob_tst = np.pad(y_prob_tst, pad_width=((0, 0), (0, Ko)))
    perm_labels = ind_labels + ood_labels

    cm = sklearn.metrics.confusion_matrix(
        y_true_tst, y_prob_tst.argmax(-1), labels=list(range(Ki + Ko))
    )
    disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=perm_labels)
    disp.plot()
    fp = os.path.join(config.experiment_path, f"cm_tst_e{step}.png")
    plt.savefig(fp)
    plt.close()
    figs["tst/cm"] = wandb.Image(Image.open(fp))

    tqdm.write(f"Epoch {step}: {metrics}")

    join = lambda x: ",".join(map(perm_labels.__getitem__, x))
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
        prefix_stats: dict = stats[prefix]
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
        figs[f"{prefix}/roc"] = wandb.plot.roc_curve(target, values_2d, LABELS)
        figs[f"{prefix}/pr"] = wandb.plot.pr_curve(target, values_2d, LABELS)
        figs[f"{prefix}/auroc"] = sklearn.metrics.roc_auc_score(target, values)

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
    for epoch in range(1, config.train_epochs + 1):
        epoch_stats: dict = {}

        # train
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
