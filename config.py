import logging
import os
import sys
from typing import List, Tuple

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders


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
        image_chw: Tuple[int, int, int],
        patch_hw: Tuple[int, int],
        manifold_c: int,
        batch_size: int,
        optim_lr: float,
        optim_m: float,
        train_epochs: int,
        exc_dry_run: bool,
        exc_resume: bool,
        input_chw: Tuple[int, int, int],
        dataset_info: dict,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        coupling_network_config: dict,
        conv1x1_config: dict,
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
        self.patch_hw = patch_hw
        self.manifold_c = manifold_c
        self.batch_size = batch_size
        self.optim_lr = optim_lr
        self.optim_m = optim_m
        self.train_epochs = train_epochs
        self.exc_dry_run = exc_dry_run
        self.exc_resume = exc_resume
        self.input_chw = input_chw
        self.dataset_info = dataset_info
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.coupling_network_config = coupling_network_config
        self.conv1x1_config = conv1x1_config
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
    patch_hw = (int(getenv("PATCH_H")), int(getenv("PATCH_W")))
    manifold_c = int(getenv("MANIFOLD_C"))
    batch_size = int(getenv("BATCH_SIZE"))
    optim_lr = float(getenv("OPTIM_LR"))
    optim_m = float(getenv("OPTIM_M"))
    train_epochs = int(getenv("TRAIN_EPOCHS"))
    exc_dry_run = bool(int(getenv("EXC_DRY_RUN")))
    exc_resume = bool(int(getenv("EXC_RESUME")))
    label_type = getenv("LABEL_TYPE")

    # image dims (get from the data loader)
    image_chw = get_dataset_chw(dataset_name)

    # input dims for model
    input_chw = (
        image_chw[-3] * patch_hw[-2] * patch_hw[-1],
        image_chw[-2] // patch_hw[-2],
        image_chw[-1] // patch_hw[-1],
    )

    # data loader
    train_loader, test_loader = get_dataset_loaders(
        dataset_name,
        batch_size_train=batch_size,
        batch_size_test=batch_size,
        data_root=data_dir,
        cv_k=cv_k,
        cv_folds=cv_folds,
        cv_mode=cv_mode,
        label_type=label_type,
    )
    dataset_info = get_dataset_info(
        dataset_name,
        data_root=data_dir,
        cv_mode=cv_mode,
        label_type=label_type,
    )
    # runtime device
    device = get_best_device()
    logging.info(f"Using device: {device}")

    # tqdm prompt
    tqdm_args = {
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
        "file": sys.stdout,
    }
    # network configuration
    coupling_network_config = {
        "num_channels": input_chw[0] // 2,
        "num_hidden_channels": input_chw[0],
        "num_hidden_layers": 1,
    }
    conv1x1_config = {"num_channels": input_chw[0]}

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
        image_chw=image_chw,
        patch_hw=patch_hw,
        manifold_c=manifold_c,
        batch_size=batch_size,
        optim_lr=optim_lr,
        optim_m=optim_m,
        train_epochs=train_epochs,
        exc_dry_run=exc_dry_run,
        exc_resume=exc_resume,
        # derived params
        input_chw=input_chw,
        dataset_info=dataset_info,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        coupling_network_config=coupling_network_config,
        conv1x1_config=conv1x1_config,
        # hardcoded params
        tqdm_args=tqdm_args,
    )
