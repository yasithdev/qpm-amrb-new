import json
import os
import shutil
import sys

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt, gridspec
from tqdm import tqdm


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

  hist_x, hist_y = np.unique(bbox_vals, return_counts=True)
  # set up a (2 x 10) plot for W samples + their contour maps
  w = 8
  h = 6
  plt.figure(figsize=(w, h))

  # create summary title
  plt.title(f"{label} : Invalid ({num_invalid}), Accepted ({num_accepted}/{num_oversize_acc}), Rejected ({num_rejected}/{num_oversize_rej})")

  plt.bar(hist_x, hist_y)
  plt.ylabel("Count")
  plt.xlabel("Contour Size")
  plt.axvline(l, color='red')
  # save figure and close
  plt.savefig(path)
  plt.close()


def main():
  load_dotenv()

  tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}

  data_dir = os.getenv("DATA_DIR")
  print(f"DATA_DIR={data_dir}")

  dataset_name = os.getenv("DS_NAME")
  print(f"DS_NAME={dataset_name}")

  dataset_path = os.path.join(data_dir, dataset_name)
  print(f"(Dataset Path)={dataset_path}")

  experiment_dir = os.getenv("EXPERIMENT_DIR")
  print(f"EXPERIMENT_DIR={experiment_dir}")

  experiment_name = os.getenv("EXPERIMENT_NAME")
  print(f"EXPERIMENT_NAME={experiment_name}")

  experiment_path = os.path.join(experiment_dir, experiment_name)
  print(f"(Experiment Path)={experiment_path}")

  img_l = int(os.getenv("L"))
  print(f"L={img_l}")

  print(f"Reading info.json")
  with open(os.path.join(dataset_path, "info.json"), "r") as f:
    labels = sorted(json.load(f)["data"])

  ds_imag_accepted_path = os.path.join(experiment_path, f"accepted.imag.npz")
  ds_mask_accepted_path = os.path.join(experiment_path, f"accepted.mask.npz")
  ds_imag_rejected_path = os.path.join(experiment_path, f"rejected.imag.npz")
  ds_mask_rejected_path = os.path.join(experiment_path, f"rejected.mask.npz")
  ds_stats_path = os.path.join(experiment_path, f"stats.npz")

  ds_imag_accepted = np.load(ds_imag_accepted_path)
  ds_mask_accepted = np.load(ds_mask_accepted_path)
  ds_imag_rejected = np.load(ds_imag_rejected_path)
  ds_mask_rejected = np.load(ds_mask_rejected_path)
  ds_stats = np.load(ds_stats_path, allow_pickle=True)

  print("Generating visualizations...")

  # initialize visualization directories
  vis_dir = os.path.join(experiment_path, "vis")
  vis_dir_accepted = os.path.join(vis_dir, "accepted")
  vis_dir_rejected = os.path.join(vis_dir, "rejected")

  shutil.rmtree(vis_dir_accepted, ignore_errors=True)
  shutil.rmtree(vis_dir_rejected, ignore_errors=True)

  os.makedirs(vis_dir_accepted, exist_ok=True)
  os.makedirs(vis_dir_rejected, exist_ok=True)

  for label in tqdm(labels, desc=f'Generating Statistics: ', **tqdm_args):
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
      img_l,
      os.path.join(vis_dir, f"stats-{label}.png")
    )
  print("DONE")


if __name__ == '__main__':
  main()
