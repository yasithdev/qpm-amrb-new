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

        aug_pool = np.concatenate(
            [
                np.rot90(source_images, k=1, axes=(1, 2)),
                np.rot90(source_images, k=2, axes=(1, 2)),
                np.rot90(source_images, k=3, axes=(1, 2)),
            ],
            axis=0,
        )

        aug_mask = np.ones(aug_size, dtype=bool)

        # select unselected images from pool and add to dataset
        for _ in range(target_count - source_count):
            while True:
                rand_idx = np.random.randint(0, aug_size)
                if aug_mask[rand_idx] == 1:
                    break
            aug_mask[rand_idx] = 0
            aug_data.append(aug_pool[rand_idx])
        target_images = np.concatenate([source_images, np.array(aug_data)], axis=0)

    assert target_images.shape[0] == target_count
    return target_images


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
        experiment_dir, "3_train_test_split", dataset_name
    )
    print(f"\n(Data Source Experiment Path)={data_source_experiment_path}")

    experiment_name = "4_augment"
    print(f"\n(Experiment Name)={experiment_name}")

    experiment_path = os.path.join(experiment_dir, experiment_name, dataset_name)
    print(f"(Experiment Path)={experiment_path}")

    # initialize experiment directories
    shutil.rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path, exist_ok=True)

    for k in range(folds):

        print(f"\nFold {k+1} of {folds}\n")

        print(f"\nLoading {dataset_name}:")
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
        print("augmenting train/test splits")
        aug_trn_data = {}
        aug_tst_data = {}
        for target in tqdm(targets, **tqdm_args):
            aug_trn_data[target] = shuffle_and_augment(
                trn_data[target], target_count_trn
            )
            aug_tst_data[target] = shuffle_and_augment(
                tst_data[target], target_count_tst
            )

        print("Writing to file...")
        np.savez_compressed(
            os.path.join(experiment_path, f"train_k{k}.npz"), **aug_trn_data
        )
        np.savez_compressed(
            os.path.join(experiment_path, f"test_k{k}.npz"), **aug_tst_data
        )
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
