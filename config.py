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
        checkpoint_metric: str,
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
        self.checkpoint_metric = checkpoint_metric
        self.image_chw: Tuple[int, int, int] | None = None
        self.labels: List[str] | None = None
        self.datamodule: LightningDataModule | None = None
        self.ood: List[int] = []
        for k in self.ood_k.split(':'):
            if len(k) > 0:
                self.ood.append(int(k))

    def load_data(
        self,
        *,
        dataset_name: str | None = None,
        data_dir: str | None = None,
        batch_size: int | None = None,
        ood: list[int] | None = None,
    ) -> None:
        """
        Load data modules.

        This function uses pre-configured parameters by default.
        If you need to override anything in code, specify them in the function args.

        Args:
            dataset_name (str, optional): Name of the dataset.
            data_dir (str, optional): Location of the data directory.
            batch_size (int, optional): Batch size to use.
            ood (list[int], optional): Targets to treat as OOD.
        """
        from datasets import get_data
        dm = get_data(
            dataset_name or self.dataset_name,
            data_dir or self.data_dir,
            batch_size or self.batch_size,
            ood or self.ood,
        )
        self.datamodule = dm
        self.labels = dm.targets
        self.image_chw = dm.shape

    def get_model(self):
        from models import get_model
        from datasets.transforms import reindex_for_ood
        assert self.labels is not None
        assert self.image_chw is not None
        cat_k = len(self.get_ind_labels())
        return get_model(
            self.image_chw,
            self.model_name,
            reindex_for_ood(self.labels, self.ood),
            cat_k,
            self.manifold_d,
            self.optim_lr
        )

    def get_ind_labels(self) -> List[str]:
        assert self.labels is not None
        ind_labels = [x for i, x in enumerate(self.labels) if i not in self.ood]
        return ind_labels

    def get_ood_labels(self) -> List[str]:
        assert self.labels is not None
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
    matplotlib.rcParams["figure.figsize"] = [16, 12]

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
    checkpoint_metric = getenv("CHECKPOINT_METRIC")

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
        checkpoint_metric=checkpoint_metric,
    )
