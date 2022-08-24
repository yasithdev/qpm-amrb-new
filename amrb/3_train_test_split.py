import os
import shutil
import sys

import einops
import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv


def shuffle_and_augment(
  source_images: np.array,
  target_count: int,
):

  source_count = source_images.shape[0]

  if source_count >= target_count:

    target_images = source_images[:target_count]
    np.random.shuffle(target_images)
  
  else:

    aug_data = []
    aug_size = source_count * 3

    aug_pool = np.concatenate([
      np.rot90(source_images, k=1, axes=(1, 2)),
      np.rot90(source_images, k=2, axes=(1, 2)),
      np.rot90(source_images, k=3, axes=(1, 2))
    ], axis=0)

    aug_mask = np.ones(aug_size, dtype=bool)

    # select unselected images from pool and add to dataset
    for _ in range(target_count - source_count):
      while True:
        rand_idx = np.random.randint(0, aug_size)
        if aug_mask[rand_idx] == 1:
          break
      aug_mask[rand_idx] = 0
      aug_data.append(aug_pool[rand_idx])
    target_images = np.concatenate([
      source_images,
      np.array(aug_data)
    ], axis=0)

  assert target_images.shape[0] == target_count
  return target_images


def main():

  data_dir = os.getenv("DATA_DIR")
  print(f"DATA_DIR={data_dir}")

  dataset_name = os.getenv("DS_NAME")
  print(f"DS_NAME={dataset_name}")

  experiment_dir = os.getenv("EXPERIMENT_DIR")
  print(f"EXPERIMENT_DIR={experiment_dir}")

  img_l = int(os.getenv("L"))
  print(f"L={img_l}")

  data_source_experiment_path = os.path.join(experiment_dir, "1_preprocessing", dataset_name)
  print(f"\n(Data Source Experiment Path)={data_source_experiment_path}")

  experiment_name = "3_train_test_split"
  print(f"\n(Experiment Name)={experiment_name}")

  experiment_path = os.path.join(experiment_dir, experiment_name, dataset_name)
  print(f"(Experiment Path)={experiment_path}")

  # initialize experiment directories
  shutil.rmtree(experiment_path, ignore_errors=True)
  os.makedirs(experiment_path, exist_ok=True)

  print(f"\nLoading {dataset_name}:")
  dataset = np.load(os.path.join(data_source_experiment_path, "accepted.imag.npz"))
  split_fraction = 0.8

  targets = sorted(dataset.keys())

  trn_data = {}
  tst_data = {}

  print("Performing filtering + train/test split")
  for target in tqdm(targets, **tqdm_args):
    images = dataset[target].astype(np.float32)
    # num_pre_cleanup_images = images.shape[0]

    # remove outliers
    images = images[np.min(images, axis=(1, 2)) >= 0]
    i_max = np.max(images, axis=(1, 2))
    mu_max, std_max = np.mean(i_max), np.std(i_max)
    images = images[np.max(images, axis=(1, 2)) <= mu_max + 2 * std_max]
    num_images = images.shape[0]
    # print("cleanup:", num_pre_cleanup_images, "->", num_images)

    pivot = int(num_images * split_fraction)
    trn_data[target] = images[:pivot]
    tst_data[target] = images[pivot:]

  target_count_trn = min(
    max([trn_data[x].shape[0] for x in targets]),
    min([trn_data[x].shape[0] for x in targets]) * 4
  )
  target_count_tst = min(
    max([tst_data[x].shape[0] for x in targets]),
    min([tst_data[x].shape[0] for x in targets]) * 4
  )

  # augment data into a uniform distribution
  print("augmenting train/test splits")
  aug_trn_data = {}
  aug_tst_data = {}
  for target in tqdm(targets, **tqdm_args):
    aug_trn_data[target] = shuffle_and_augment(trn_data[target], target_count_trn)
    aug_tst_data[target] = shuffle_and_augment(tst_data[target], target_count_tst)

  # combine data into array
  print("generating x,y data for train and test sets")
  X_TRAIN = []
  Y_TRAIN = []
  X_TEST = []
  Y_TEST = []

  for i, target in enumerate(tqdm(targets, **tqdm_args)):
    # create train dataset
    x_trn, y_trn = aug_trn_data[target].astype(np.float32), np.empty(shape=(target_count_trn, 1), dtype=np.int32)
    y_trn.fill(i)
    X_TRAIN.append(x_trn)
    Y_TRAIN.append(y_trn)
    # create test dataset
    x_tst, y_tst = aug_tst_data[target].astype(np.float32), np.empty(shape=(target_count_tst, 1), dtype=np.int32)
    y_tst.fill(i)
    X_TEST.append(x_tst)
    Y_TEST.append(y_tst)

  print("rearranging + writing data to file")
  px = 'n l h w -> (n l) h w 1'
  py = 'n l 1 -> (n l)'
  jobs = [
    (X_TRAIN, px, "trn_x.npy"),
    (Y_TRAIN, py, "trn_y.npy"),
    (X_TEST, px, "tst_x.npy"),
    (Y_TEST, py, "tst_y.npy")
  ]
  for (d, p, fp) in tqdm(jobs, **tqdm_args):
    d = np.stack(d, axis=1)
    d = einops.rearrange(d, p)
    np.save(os.path.join(experiment_path, fp), d)


if __name__ == '__main__':

  # initialize the RNG deterministically
  np.random.seed(42)

  load_dotenv()

  tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}

  main()
