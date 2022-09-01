import logging
import os
import sys
from typing import Tuple

from dotenv import load_dotenv

from data_loaders import DataLoaderFunction, get_data_loader
from dflows.nn_util import get_best_device


class Config:
    def __init__(
        self,
        log_level: str,
        crossval_k: int,
        data_dir: str,
        dataset_name: str,
        experiment_dir: str,
        saved_model_dir: str,
        image_chw: Tuple[int, int, int],
        patch_hw: Tuple[int, int],
        manifold_c: int,
        batch_size: int,
        optim_lr: float,
        optim_m: float,
        train_epochs: int,
        exc_dry_run: bool,
        exc_resume: bool,
        exc_ood_mode: bool,
        input_chw: Tuple[int, int, int],
        data_loader: DataLoaderFunction,
        device: str,
        coupling_network_config: dict,
        conv1x1_config: dict,
        tqdm_args: dict,
    ) -> None:
        self.log_level = log_level
        self.crossval_k = crossval_k
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.experiment_dir = experiment_dir
        self.saved_model_dir = saved_model_dir
        self.image_chw = image_chw
        self.patch_hw = patch_hw
        self.manifold_c = manifold_c
        self.batch_size = batch_size
        self.optim_lr = optim_lr
        self.optim_m = optim_m
        self.train_epochs = train_epochs
        self.exc_dry_run = exc_dry_run
        self.exc_resume = exc_resume
        self.exc_ood_mode = exc_ood_mode
        self.input_chw = input_chw
        self.data_loader = data_loader
        self.device = device
        self.coupling_network_config = coupling_network_config
        self.conv1x1_config = conv1x1_config
        self.tqdm_args = tqdm_args


def getenv(key: str, default=None) -> str:
    val = os.getenv(key, default)
    val = os.path.expanduser(val)
    val = os.path.expandvars(val)
    logging.info(f"{key}={val}")
    return val


def load_config() -> Config:

    load_dotenv()

    log_level = os.getenv("LOG_LEVEL")
    logging.basicConfig(level=log_level)
    logging.info(f"LOG_LEVEL={log_level}")

    crossval_k = int(getenv("CROSSVAL_K"))
    data_dir = getenv("DATA_DIR")
    dataset_name = getenv("DS_NAME")
    experiment_dir = getenv("EXPERIMENT_DIR")
    saved_model_dir = getenv("SAVED_MODEL_DIR")
    image_chw = (int(getenv("IMAGE_C")), int(getenv("IMAGE_H")), int(getenv("IMAGE_W")))
    patch_hw = (int(getenv("PATCH_H")), int(getenv("PATCH_W")))
    manifold_c = int(getenv("MANIFOLD_C"))
    batch_size = int(getenv("BATCH_SIZE"))
    optim_lr = float(getenv("OPTIM_LR"))
    optim_m = float(getenv("OPTIM_M"))
    train_epochs = int(getenv("TRAIN_EPOCHS"))
    exc_dry_run = bool(int(getenv("EXC_DRY_RUN")))
    exc_resume = bool(int(getenv("EXC_RESUME")))
    exc_ood_mode = bool(int(getenv("EXC_OOD_MODE")))

    # input dims for model
    input_chw = (
        image_chw[-3] * patch_hw[-2] * patch_hw[-1],
        image_chw[-2] // patch_hw[-2],
        image_chw[-1] // patch_hw[-1],
    )
    # data loader
    data_loader = get_data_loader(
        dataset_name,
        batch_size_train=batch_size,
        batch_size_test=batch_size,
        data_root=data_dir,
        ood_mode=exc_ood_mode,
        crossval_k=crossval_k
    )
    # runtime device
    device = get_best_device()
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
        crossval_k=crossval_k,
        data_dir=data_dir,
        dataset_name=dataset_name,
        experiment_dir=experiment_dir,
        saved_model_dir=saved_model_dir,
        image_chw=image_chw,
        patch_hw=patch_hw,
        manifold_c=manifold_c,
        batch_size=batch_size,
        optim_lr=optim_lr,
        optim_m=optim_m,
        train_epochs=train_epochs,
        exc_dry_run=exc_dry_run,
        exc_resume=exc_resume,
        exc_ood_mode=exc_ood_mode,
        # derived params
        input_chw=input_chw,
        data_loader=data_loader,
        device=device,
        coupling_network_config=coupling_network_config,
        conv1x1_config=conv1x1_config,
        # hardcoded params
        tqdm_args=tqdm_args,
    )
