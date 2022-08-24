import os
import shutil
import sys

import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv


def main(folds: int):

    data_dir = os.getenv("DATA_DIR")
    print(f"DATA_DIR={data_dir}")

    dataset_name = os.getenv("DS_NAME")
    print(f"DS_NAME={dataset_name}")

    experiment_dir = os.getenv("EXPERIMENT_DIR")
    print(f"EXPERIMENT_DIR={experiment_dir}")

    img_l = int(os.getenv("L"))
    print(f"L={img_l}")

    data_source_experiment_path = os.path.join(
        experiment_dir, "1_preprocessing", dataset_name
    )
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
    targets = sorted(dataset.keys())

    # filter outliers
    filtered_dataset = {}
    print("Filtering outliers:")
    for target in targets:

        source_images = dataset[target]

        # remove outliers
        target_images = source_images
        target_images = target_images[np.min(target_images, axis=(1, 2)) >= 0]
        i_max = np.max(target_images, axis=(1, 2))
        mu_max, std_max = np.mean(i_max), np.std(i_max)
        target_images = target_images[
            np.max(target_images, axis=(1, 2)) <= mu_max + 2 * std_max
        ]

        filtered_dataset[target] = target_images
        print(f"{target}: {source_images.shape[0]} -> {target_images.shape[0]}")

    split_fraction = 0.8
    for k in range(folds):

        print(f"\nFold {k+1} of {folds}\n")

        trn_data = {}
        tst_data = {}

        print("performing train/test split")
        for target in tqdm(targets, **tqdm_args):

            # get data
            images = filtered_dataset[target]
            num_images = images.shape[0]

            # roll images for k-fold split
            shift = int(num_images * k / folds)
            images = np.roll(images, shift, axis=0)

            shift = int(num_images * split_fraction)
            trn_data[target] = images[:shift]
            tst_data[target] = images[shift:]

        print("Writing to file...")
        np.savez_compressed(
            os.path.join(experiment_path, f"train_k{k}.npz"), **trn_data
        )
        np.savez_compressed(os.path.join(experiment_path, f"test_k{k}.npz"), **tst_data)
        print("DONE")


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)

    load_dotenv()

    tqdm_args = {
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
        "file": sys.stdout,
    }

    folds = 10
    main(folds)
