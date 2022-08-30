import logging
import os
import shutil

import numpy as np
from tqdm import tqdm

from amrb_util import load_config


def main(config: dict):

    data_source_experiment_path = os.path.join(
        config.get("experiment_dir"), "1_preprocessing", config.get("dataset_name")
    )
    logging.info(f"(Data Source Experiment Path)={data_source_experiment_path}")

    experiment_name = "3_train_test_split"
    logging.info(f"(Experiment Name)={experiment_name}")

    experiment_path = os.path.join(
        config.get("experiment_dir"), experiment_name, config.get("dataset_name")
    )
    logging.info(f"(Experiment Path)={experiment_path}")

    # initialize experiment directories
    shutil.rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path, exist_ok=True)

    logging.info(f"Loading {config.get('dataset_name')}:")
    dataset = np.load(os.path.join(data_source_experiment_path, "accepted.imag.npz"))
    targets = sorted(dataset.keys())

    # filter outliers
    filtered_dataset = {}
    logging.info("Filtering outliers:")
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
        logging.info(f"{target}: {source_images.shape[0]} -> {target_images.shape[0]}")

    split_fraction = 0.8
    for k in range(config.get("folds")):

        logging.info(f"Fold {k+1} of {config.get('folds')}")

        trn_data = {}
        tst_data = {}

        logging.info("performing train/test split")
        for target in tqdm(targets, **config.get("tqdm_args")):

            # get data
            images = filtered_dataset[target]
            num_images = images.shape[0]

            # roll images for k-fold split
            shift = int(num_images * k / config.get("folds"))
            images = np.roll(images, shift, axis=0)

            shift = int(num_images * split_fraction)
            trn_data[target] = images[:shift]
            tst_data[target] = images[shift:]

        trn_data_path = os.path.join(experiment_path, f"train_k{k}.npz")
        logging.info(f"Writing: {trn_data_path}")
        np.savez_compressed(trn_data_path, **trn_data)
        logging.info(f"Written: {trn_data_path}")

        tst_data_path = os.path.join(experiment_path, f"test_k{k}.npz")
        logging.info(f"Writing: {tst_data_path}")
        np.savez_compressed(tst_data_path, **tst_data)
        logging.info(f"Written: {tst_data_path}")

        logging.info("DONE")


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    config = load_config()
    main(config)
