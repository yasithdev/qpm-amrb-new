import logging
import os
import shutil
import sys
from typing import Tuple, Optional, List

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader


def get_best_device():

    from torch.backends import cuda, mps

    # check for cuda
    if cuda.is_built() and torch.cuda.is_available():
        return "cuda"
    # check for mps
    # if mps.is_built() and mps.is_available():
    #     return "mps"
    return "cpu"


class Config:
    def __init__(
        self,
        log_level: str,
        cv_k: int,
        cv_folds: int,
        cv_mode: str,
        data_dir: str,
        dataset_name: str,
        model_name: str,
        experiment_base: str,
        manifold_d: int,
        batch_size: int,
        optim_lr: float,
        optim_m: float,
        train_epochs: int,
        exc_resume: bool,
        dataset_info: Optional[dict],
        image_chw: Optional[Tuple[int, int, int]],
        train_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        labels: Optional[List],
        device: str,
        tqdm_args: dict,
    ) -> None:
        self.log_level = log_level
        self.cv_k = cv_k
        self.cv_folds = cv_folds
        self.cv_mode = cv_mode
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.experiment_base = experiment_base
        self.image_chw = image_chw
        self.manifold_d = manifold_d
        self.batch_size = batch_size
        self.optim_lr = optim_lr
        self.optim_m = optim_m
        self.train_epochs = train_epochs
        self.exc_resume = exc_resume
        self.dataset_info = dataset_info
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels = labels
        self.device = device
        self.tqdm_args = tqdm_args

        dir_1 = self.dataset_name
        dir_2 = f"{self.cv_mode}-N{self.cv_folds}-K{self.cv_k}-M{self.manifold_d}"
        self.experiment_path = os.path.join(self.experiment_base, dir_1, dir_2)
        self.run_name = f"{dir_1}-{dir_2}"
        self.run_config = {
            "dataset": self.dataset_name,
            "cv_mode": self.cv_mode,
            "cv_folds": self.cv_folds,
            "cv_k": self.cv_k,
            "model": self.model_name,
            "manifold_d": self.manifold_d,
        }

        if not self.exc_resume:
            shutil.rmtree(self.experiment_path, ignore_errors=True)
        os.makedirs(self.experiment_path, exist_ok=True)

    def init_labels(
        self,
    ) -> None:
        assert self.train_loader
        assert self.test_loader

        train_labels = list(self.train_loader.dataset.labels)  # type: ignore
        test_labels = list(self.test_loader.dataset.labels)  # type: ignore

        if self.cv_mode == "leave-out":
            logging.info(f"Labels (train): {train_labels}")
            logging.info(f"Labels (test): {test_labels}")
            assert set(train_labels).isdisjoint(test_labels)
            labels = [*train_labels, *test_labels]
        else:
            assert set(train_labels) == set(test_labels)
            labels = train_labels
            logging.info(f"Labels (train, test): {train_labels}")

        self.labels = labels


def getenv(key: str, default="") -> str:
    val = os.getenv(key, default)
    val = os.path.expanduser(val)
    val = os.path.expandvars(val)
    logging.info(f"{key}={val}")
    return val


def load_config() -> Config:

    load_dotenv()

    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=log_level)
    logging.info(f"LOG_LEVEL={log_level}")

    data_dir = getenv("DATA_DIR")
    dataset_name, cv_k = getenv("DATASET_NAME").rsplit(".", maxsplit=1)
    cv_k = int(cv_k)
    cv_folds = int(getenv("CV_FOLDS"))
    cv_mode = getenv("CV_MODE")
    model_name = getenv("MODEL_NAME")
    experiment_base = getenv("EXPERIMENT_BASE")
    manifold_d = int(getenv("MANIFOLD_D"))
    batch_size = int(getenv("BATCH_SIZE"))
    optim_lr = float(getenv("OPTIM_LR"))
    optim_m = float(getenv("OPTIM_M"))
    train_epochs = int(getenv("TRAIN_EPOCHS"))
    exc_resume = bool(int(getenv("EXC_RESUME")))

    # runtime device
    device = get_best_device()
    logging.info(f"Using device: {device}")

    # tqdm prompt
    tqdm_args = {
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
        "file": sys.stdout,
    }

    return Config(
        # env params
        log_level=log_level,
        cv_k=cv_k,
        cv_folds=cv_folds,
        cv_mode=cv_mode,
        data_dir=data_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        experiment_base=experiment_base,
        manifold_d=manifold_d,
        batch_size=batch_size,
        optim_lr=optim_lr,
        optim_m=optim_m,
        train_epochs=train_epochs,
        exc_resume=exc_resume,
        # derived params
        dataset_info=None,
        image_chw=None,
        train_loader=None,
        test_loader=None,
        labels=None,
        device=device,
        # hardcoded params
        tqdm_args=tqdm_args,
    )
