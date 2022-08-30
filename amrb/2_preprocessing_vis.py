import json
import logging
import os
import shutil

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from tqdm import tqdm

from amrb_util import load_config


def plot_image_and_contour(
    samples: list,
    cmaps: list,
    title: str,
    path: str,
):
    # set up a (2 x 10) plot for W samples + their contour maps
    w = 10
    h = 2
    plt.figure(figsize=(w + 1, h + 1))
    plt.suptitle(f"{title} (Total: {len(samples)})")
    gs = gridspec.GridSpec(
        2,
        10,
        wspace=0.05,
        hspace=0.05,
        top=1.0 - 0.5 / (h + 1),
        bottom=0.5 / (h + 1),
        left=0.5 / (w + 1),
        right=1 - 0.5 / (w + 1),
    )

    # add samples to plot, if any
    w = min(len(samples), 10)
    indices = np.random.choice(np.arange(len(samples)), w, replace=False)
    for r in range(w):
        idx = indices[r]
        # top row
        ax = plt.subplot(gs[0, r])
        ax.imshow(samples[idx], cmap="gray")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # bottom row
        ax = plt.subplot(gs[1, r])
        ax.imshow(cmaps[idx], cmap="gray")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # save figure and close
    plt.savefig(path)
    plt.close()


def plot_stats(
    stats: np.ndarray,
    label: str,
    l: int,
    path: str,
):
    num_invalid = stats[f"{label}_cnum_0"]
    num_accepted = stats[f"{label}_cnum_Y_size_Y"]
    num_oversize_acc = stats[f"{label}_cnum_Y_size_N"]
    num_rejected = stats[f"{label}_cnum_N_size_Y"]
    num_oversize_rej = stats[f"{label}_cnum_N_size_N"]
    cnum_vals = stats[f"{label}_cnum_vals"]
    bbox_vals = stats[f"{label}_bbox_vals"]

    # set up a (2 x 10) plot for W samples + their contour maps
    w = 8
    h = 6
    plt.figure(figsize=(w, h))

    # create summary title
    plt.title(
        f"{label} : Invalid ({num_invalid}), Accepted ({num_accepted}/{num_oversize_acc}), Rejected ({num_rejected}/{num_oversize_rej})"
    )

    # find unique cnum_vals
    cnum_uniq = np.unique(cnum_vals)

    for cnum in cnum_uniq:
        cond = cnum_vals == cnum
        hist_x, hist_y = np.unique(bbox_vals[cond], return_counts=True)
        plt.bar(hist_x, hist_y, label=f"Contours: {cnum}")

    plt.legend()
    plt.ylabel("Count")
    plt.xlabel("Contour Size")
    plt.axvline(l, color="red")
    # save figure and close
    plt.savefig(path)
    plt.close()


def main(config: dict):

    dataset_path = os.path.join(config.get("data_dir"), config.get("dataset_name"))
    logging.info(f"(Dataset Path)={dataset_path}")

    data_source_experiment_path = os.path.join(
        config.get("experiment_dir"), "1_preprocessing", config.get("dataset_name")
    )
    logging.info(f"(Data Source Experiment Path)={data_source_experiment_path}")

    experiment_name = "2_preprocessing_vis"
    logging.info(f"(Experiment Name)={experiment_name}")

    experiment_path = os.path.join(
        config.get("experiment_dir"), experiment_name, config.get("dataset_name")
    )
    logging.info(f"(Experiment Path)={experiment_path}")

    info_json_path = os.path.join(dataset_path, "info.json")
    logging.info(f"Reading: {info_json_path}")
    with open(info_json_path, "r") as f:
        labels = sorted(json.load(f)["data"])

    ds_imag_accepted_path = os.path.join(
        data_source_experiment_path, f"accepted.imag.npz"
    )
    ds_mask_accepted_path = os.path.join(
        data_source_experiment_path, f"accepted.mask.npz"
    )
    ds_imag_rejected_path = os.path.join(
        data_source_experiment_path, f"rejected.imag.npz"
    )
    ds_mask_rejected_path = os.path.join(
        data_source_experiment_path, f"rejected.mask.npz"
    )
    ds_stats_path = os.path.join(data_source_experiment_path, f"stats.npz")

    ds_imag_accepted = np.load(ds_imag_accepted_path)
    ds_mask_accepted = np.load(ds_mask_accepted_path)
    ds_imag_rejected = np.load(ds_imag_rejected_path)
    ds_mask_rejected = np.load(ds_mask_rejected_path)
    ds_stats = np.load(ds_stats_path, allow_pickle=True)

    # initialize visualization directories
    vis_dir_accepted = os.path.join(experiment_path, "accepted")
    vis_dir_rejected = os.path.join(experiment_path, "rejected")
    vis_dir_stats = os.path.join(experiment_path, "stats")

    shutil.rmtree(vis_dir_accepted, ignore_errors=True)
    shutil.rmtree(vis_dir_rejected, ignore_errors=True)
    shutil.rmtree(vis_dir_stats, ignore_errors=True)

    os.makedirs(vis_dir_accepted, exist_ok=True)
    os.makedirs(vis_dir_rejected, exist_ok=True)
    os.makedirs(vis_dir_stats, exist_ok=True)

    for label in tqdm(
        labels, desc=f"Generating Visualizations: ", **config.get("tqdm_args")
    ):
        # plot 1 - examples from accepted category
        plot_image_and_contour(
            ds_imag_accepted[label],
            ds_mask_accepted[label],
            f"Single Object: {label}",
            os.path.join(vis_dir_accepted, f"{label}.png"),
        )

        # plot 2 - examples from rejected category
        plot_image_and_contour(
            ds_imag_rejected[label],
            ds_mask_rejected[label],
            f"Multiple Objects: {label}",
            os.path.join(vis_dir_rejected, f"{label}.png"),
        )

        # plot 3 - statistics
        plot_stats(
            ds_stats,
            label,
            config.get("img_l"),
            os.path.join(vis_dir_stats, f"{label}.png"),
        )
    logging.info("DONE")


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    config = load_config()
    main(config)
