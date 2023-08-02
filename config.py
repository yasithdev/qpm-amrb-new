import logging
import os
import shutil
import sys
from typing import List, Optional, Tuple

import torch
from dotenv import load_dotenv
from lightning.pytorch import LightningDataModule


def get_best_device():
    from torch.backends import cuda, mps

    # check for cuda
    if cuda.is_built() and torch.cuda.is_available():
        return "cuda"
    # check for mps
    if mps.is_built() and mps.is_available():
        return "mps"
    return "cpu"


class Config:
    def __init__(
        self,
        log_level: str,
        ood_k: str,
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
        device: str,
        tqdm_args: dict,
        #
        image_chw: Optional[Tuple[int, int, int]] = None,
        labels: Optional[List[str]] = None,
        datamodule: Optional[LightningDataModule] = None
    ) -> None:
        self.log_level = log_level
        self.ood_k = ood_k
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.experiment_base = experiment_base
        self.manifold_d = manifold_d
        self.batch_size = batch_size
        self.optim_lr = optim_lr
        self.optim_m = optim_m
        self.train_epochs = train_epochs
        self.exc_resume = exc_resume
        self.device = device
        self.tqdm_args = tqdm_args

        self.ood: List[int] = []
        if len(self.ood_k) > 0:
            self.ood.append(int(self.ood_k))

        fragments = []
        fragments.append(self.dataset_name)
        fragments.append(f"M{self.manifold_d}")
        if len(self.ood_k) > 0:
            fragments.append(f"OOD{self.ood_k}")
        self.run_name = "-".join(fragments)
        self.experiment_path = os.path.join(self.experiment_base, self.run_name)
        self.run_config = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "dim_u": self.manifold_d,
            "ood_k": self.ood_k,
        }

        if not self.exc_resume:
            shutil.rmtree(self.experiment_path, ignore_errors=True)
        os.makedirs(self.experiment_path, exist_ok=True)

        self.image_chw = image_chw
        self.labels = labels
        self.datamodule = datamodule

    def get_ind_labels(self) -> List[str]:
        assert self.labels
        ind_labels = [x for i, x in enumerate(self.labels) if i not in self.ood]
        return ind_labels

    def get_ood_labels(self) -> List[str]:
        assert self.labels
        ood_labels = [x for i, x in enumerate(self.labels) if i in self.ood]
        return ood_labels

    def print_labels(
        self,
    ) -> None:
        logging.info(f"Labels (train, test): {self.get_ind_labels()}")
        logging.info(f"Labels (ood): {self.get_ood_labels()}")


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

    ood_k = getenv("OOD_K")
    data_dir = getenv("DATA_DIR")
    dataset_name = getenv("DATASET_NAME")
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
        log_level=log_level,
        ood_k=ood_k,
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
        device=device,
        tqdm_args=tqdm_args,
    )
