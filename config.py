import logging
import os
import sys
from typing import Tuple, Optional

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from datasets import get_dataset_chw


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
        label_type: str,
        data_dir: str,
        dataset_name: str,
        model_name: str,
        experiment_dir: str,
        manifold_c: int,
        batch_size: int,
        optim_lr: float,
        optim_m: float,
        train_epochs: int,
        exc_dry_run: bool,
        exc_resume: bool,
        dataset_info: Optional[dict],
        image_chw: Optional[Tuple[int, int, int]],
        train_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: str,
        tqdm_args: dict,
    ) -> None:
        self.log_level = log_level
        self.cv_k = cv_k
        self.cv_folds = cv_folds
        self.cv_mode = cv_mode
        self.label_type = label_type
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.experiment_dir = experiment_dir
        self.image_chw = image_chw
        self.manifold_c = manifold_c
        self.batch_size = batch_size
        self.optim_lr = optim_lr
        self.optim_m = optim_m
        self.train_epochs = train_epochs
        self.exc_dry_run = exc_dry_run
        self.exc_resume = exc_resume
        self.dataset_info = dataset_info
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.tqdm_args = tqdm_args


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
    experiment_dir = getenv("EXPERIMENT_DIR")
    manifold_c = int(getenv("MANIFOLD_C"))
    batch_size = int(getenv("BATCH_SIZE"))
    optim_lr = float(getenv("OPTIM_LR"))
    optim_m = float(getenv("OPTIM_M"))
    train_epochs = int(getenv("TRAIN_EPOCHS"))
    exc_dry_run = bool(int(getenv("EXC_DRY_RUN")))
    exc_resume = bool(int(getenv("EXC_RESUME")))
    label_type = getenv("LABEL_TYPE")

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
        label_type=label_type,
        data_dir=data_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        experiment_dir=experiment_dir,
        manifold_c=manifold_c,
        batch_size=batch_size,
        optim_lr=optim_lr,
        optim_m=optim_m,
        train_epochs=train_epochs,
        exc_dry_run=exc_dry_run,
        exc_resume=exc_resume,
        # derived params
        dataset_info=None,
        image_chw=None,
        train_loader=None,
        test_loader=None,
        device=device,
        # hardcoded params
        tqdm_args=tqdm_args,
    )
