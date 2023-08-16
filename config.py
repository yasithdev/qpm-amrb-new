import logging
import os
from typing import Tuple, List
from dotenv import load_dotenv
from lightning.pytorch import LightningDataModule


class Config:
    def __init__(
        self,
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
    ) -> None:
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
        self.image_chw: Tuple[int, int, int] | None = None
        self.labels: List[str] | None = None
        self.datamodule: LightningDataModule | None = None
        self.ood: List[int] = []
        if len(self.ood_k) > 0:
            self.ood.append(int(self.ood_k))

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

    def as_dict(
        self,
    ) -> dict[str, int | str | float]:
        return dict(
            ood_k=self.ood_k,
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            experiment_base=self.experiment_base,
            manifold_d=self.manifold_d,
            batch_size=self.batch_size,
            optim_lr=self.optim_lr,
            optim_m=self.optim_m,
            train_epochs=self.train_epochs,
        )


def getenv(key: str, default="") -> str:
    val = os.getenv(key, default)
    val = os.path.expanduser(val)
    val = os.path.expandvars(val)
    logging.info(f"{key}={val}")
    return val


def load_config() -> Config:
    # log_level = os.getenv("LOG_LEVEL", "INFO")
    log_level = logging.WARN
    logging.basicConfig(level=log_level)
    logging.info(f"LOG_LEVEL={log_level}")

    import matplotlib

    matplotlib.rcParams["figure.dpi"] = 200
    matplotlib.rcParams["figure.figsize"] = [9.6, 7.2]

    load_dotenv()
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

    return Config(
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
    )
