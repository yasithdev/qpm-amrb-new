import logging
import os
import shutil

import numpy as np
from tqdm import tqdm

from amrb_util import load_config


def shuffle_and_augment(
    source_images: np.array,
    target_count: int,
):

    source_count = source_images.shape[0]

    if source_count >= target_count:

        target_images = source_images[:target_count]
        np.random.shuffle(target_images)

    else:

        # define a pool of augmented images (by rotating 90, 180, and 270)
        aug_pool = np.concatenate(
            [
                np.rot90(source_images, k=1, axes=(1, 2)),
                np.rot90(source_images, k=2, axes=(1, 2)),
                np.rot90(source_images, k=3, axes=(1, 2)),
            ],
            axis=0,
        )
        # randomly pick images from aug_pool (without replacement)
        aug_idxs = np.random.choice(np.arange(len(aug_pool)), size=target_count-source_count, replace=False)
        aug_data = aug_pool[aug_idxs]

        # concatenate source images with aug_data
        target_images = np.concatenate([source_images, aug_data], axis=0)

    assert target_images.shape[0] == target_count
    return target_images


def main(config: dict):

    data_source_experiment_path = os.path.join(
        config.get("experiment_dir"), "3_train_test_split", config.get("dataset_name")
    )
    logging.info(f"(Data Source Experiment Path)={data_source_experiment_path}")

    experiment_name = "4_augment"
    logging.info(f"(Experiment Name)={experiment_name}")

    experiment_path = os.path.join(
        config.get("experiment_dir"), experiment_name, config.get("dataset_name")
    )
    logging.info(f"(Experiment Path)={experiment_path}")

    # initialize experiment directories
    shutil.rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path, exist_ok=True)

    for k in range(config.get("folds")):

        logging.info(f"Fold {k+1} of {config.get('folds')}")

        logging.info(f"Loading {config.get('dataset_name')}:")
        trn_data = np.load(os.path.join(data_source_experiment_path, f"train_k{k}.npz"))
        tst_data = np.load(os.path.join(data_source_experiment_path, f"test_k{k}.npz"))
        targets = sorted(trn_data.keys())

        target_count_trn = min(
            max([trn_data[x].shape[0] for x in targets]),
            min([trn_data[x].shape[0] for x in targets]) * 4,
        )
        target_count_tst = min(
            max([tst_data[x].shape[0] for x in targets]),
            min([tst_data[x].shape[0] for x in targets]) * 4,
        )

        # augment data into a uniform distribution
        logging.info("augmenting train/test splits")
        aug_trn_data = {}
        aug_tst_data = {}
        for target in tqdm(targets, **config.get("tqdm_args")):
            aug_trn_data[target] = shuffle_and_augment(
                trn_data[target], target_count_trn
            )
            aug_tst_data[target] = shuffle_and_augment(
                tst_data[target], target_count_tst
            )

        trn_data_path = os.path.join(experiment_path, f"train_k{k}.npz")
        logging.info(f"Writing: {trn_data_path}")
        np.savez_compressed(trn_data_path, **aug_trn_data)
        logging.info(f"Written: {trn_data_path}")

        tst_data_path = os.path.join(experiment_path, f"test_k{k}.npz")
        logging.info(f"Writing: {tst_data_path}")
        np.savez_compressed(tst_data_path, **aug_tst_data)
        logging.info(f"Written: {tst_data_path}")

        logging.info("DONE")


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    config = load_config()
    main(config)
