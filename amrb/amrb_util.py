import logging
import os
import sys

from dotenv import load_dotenv


def load_config():

    load_dotenv()

    tqdm_args = {
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
        "file": sys.stdout,
    }

    folds = 10

    log_level = os.getenv("LOG_LEVEL")
    logging.basicConfig(level=log_level)
    logging.info(f"LOG_LEVEL={log_level}")

    data_dir = os.getenv("DATA_DIR")
    logging.info(f"DATA_DIR={data_dir}")

    dataset_name = os.getenv("DS_NAME")
    logging.info(f"DS_NAME={dataset_name}")

    experiment_dir = os.getenv("EXPERIMENT_DIR")
    logging.info(f"EXPERIMENT_DIR={experiment_dir}")

    img_l = int(os.getenv("IMG_L"))
    logging.info(f"IMG_L={img_l}")

    return {
        "tqdm_args": tqdm_args,
        "folds": folds,
        "data_dir": data_dir,
        "dataset_name": dataset_name,
        "experiment_dir": experiment_dir,
        "img_l": img_l,
        "log_level": log_level,
    }
