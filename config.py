from __future__ import annotations

import argparse
import logging
import os
from typing import List, Tuple, TypeVar

import matplotlib
from lightning.pytorch import LightningDataModule

T = TypeVar("T", str, float, int)


def env(key: str, type: type[T], default: T) -> dict:
    val = type(os.environ.get(key, default))
    if isinstance(val, str):
        val = os.path.expanduser(val)
        val = os.path.expandvars(val)
    return dict(type=type, default=val)


def load_config(
    dataset_name: str = "",
    emb_name: str = "",
    model_name: str = "",
    ood: str = "",
    data_dir: str = "$HOME/datasets",
    emb_dir: str = "assets/embeddings",
    emb_dims: int = 128,
    emb_targets: int = 0,
    rand_perms: int = 0,
    batch_size: int = 64,
    optim_lr: float = 0.0005,
    optim_m: float = 0.8,
    train_epochs: int = 100,
    ckpt_metric: str = "val_loss",
    ckpt_mode: str = "min",
    patience: int = 30,
) -> Config:
    log_level = logging.WARN
    logging.basicConfig(level=log_level)
    logging.info(f"LOG_LEVEL={log_level}")
    matplotlib.rcParams["figure.figsize"] = [20, 15]
    matplotlib.rcParams["savefig.dpi"] = 150

    config = Config()

    parser = argparse.ArgumentParser(description="configuration")
    parser.add_argument("--data_dir", **env("DATA_DIR", type=str, default=data_dir))
    parser.add_argument("--dataset_name", **env("DATASET_NAME", type=str, default=dataset_name))
    parser.add_argument("--emb_dir", **env("EMB_DIR", type=str, default=emb_dir))
    parser.add_argument("--emb_name", **env("EMB_NAME", type=str, default=emb_name))
    parser.add_argument("--emb_dims", **env("EMB_DIMS", type=int, default=emb_dims))
    parser.add_argument("--emb_targets", **env("EMB_TARGETS", type=int, default=emb_targets))
    parser.add_argument("--rand_perms", **env("RAND_PERMS", type=int, default=rand_perms))
    parser.add_argument("--model_name", **env("MODEL_NAME", type=str, default=model_name))
    parser.add_argument("--batch_size", **env("BATCH_SIZE", type=int, default=batch_size))
    parser.add_argument("--optim_lr", **env("OPTIM_LR", type=float, default=optim_lr))
    parser.add_argument("--optim_m", **env("OPTIM_M", type=float, default=optim_m))
    parser.add_argument("--train_epochs", **env("TRAIN_EPOCHS", type=int, default=train_epochs))
    parser.add_argument("--ckpt_metric", **env("CKPT_METRIC", type=str, default=ckpt_metric))
    parser.add_argument("--ckpt_mode", **env("CKPT_MODE", type=str, default=ckpt_mode))
    parser.add_argument("--ood", **env("OOD", type=str, default=ood))
    parser.add_argument("--patience", **env("PATIENCE", type=int, default=patience))
    ## SSL Augmentation parameters
    # Common augmentations
    parser.add_argument("--scale", nargs=2, type=float, default=[0.2, 1.0])
    parser.add_argument("--train_supervised", type=bool, default=False)
    parser.add_argument("--temperature", default=0.5, type=float, help="Temperature used in softmax")
    # RGB augmentations
    parser.add_argument("--rgb_gaussian_blur_p", type=float, default=0, help="probability of using gaussian blur (rgb)")
    parser.add_argument("--rgb_jitter_d", type=float, default=1, help="color jitter 0.8*d, val 0.2*d (rgb)")
    parser.add_argument("--rgb_jitter_p", type=float, default=0.8, help="probability of using color jitter(rgb)")
    parser.add_argument("--rgb_contrast", type=float, default=0.2, help="value of contrast (rgb)")
    parser.add_argument("--rgb_contrast_p", type=float, default=0, help="prob of using contrast (rgb)")
    parser.add_argument("--rgb_grid_distort_p", type=float, default=0, help="probability of using grid distort (rgb)")
    parser.add_argument("--rgb_grid_shuffle_p", type=float, default=0, help="probability of using grid shuffle (rgb)")

    parser.parse_known_args(namespace=config)

    return config


class Config(argparse.Namespace):
    """

    Stores all configuration parameters and hyperparameters used in the project

    """

    model_name: str
    ood: list[int]
    data_dir: str
    dataset_name: str
    emb_dir: str
    emb_name: str
    emb_targets: int
    rand_perms: int
    emb_dims: int
    batch_size: int
    optim_lr: float
    optim_m: float
    train_epochs: int
    ckpt_metric: str
    ckpt_mode: str
    patience: int
    scale: tuple[float, float]
    train_supervised: bool
    temperature: float
    rgb_gaussian_blur_p: float
    rgb_jitter_d: float
    rgb_jitter_p: float
    rgb_contrast: float
    rgb_contrast_p: float
    rgb_grid_distort_p: float
    rgb_grid_shuffle_p: float

    expand_3ch: bool
    # data params - initialized when load_data() is called
    input_shape: Tuple[int, ...] | None = None
    grouping: List[int] | None = None
    labels: List[str] | None = None
    datamodule: LightningDataModule | None = None

    def __setattr__(self, name, value):
        if name == "ood":
            if type(value) == str:
                value = [int(x) for x in value.split(":") if x != ""]
            elif type(value) == list[str]:
                value = [int(x) for x in value]
            else:
                value = []
        elif name == "model_name":
            self.expand_3ch = value.startswith("resnet18") or value.startswith("resnet50")
        super().__setattr__(name, value)

    def load_data(
        self,
        *,
        dataset_name: str | None = None,
        emb_name: str | None = None,
        data_dir: str | None = None,
        emb_dir: str | None = None,
        batch_size: int | None = None,
        ood: list[int] | None = None,
        expand_3ch: bool | None = None,
    ) -> None:
        """
        Load data modules (from datasets of embeddings)

        This function uses pre-configured parameters by default.
        If you need to override anything in code, specify them in the function args.

        Args:
            dataset_name (str): Name of the dataset.
            emb_name (str): Name of the embedding.
            data_dir (str): Location of the directory containing dataset.
            emb_dir (str): Location of the directory containing embedding.
            batch_size (int): Batch size to use.
            ood (list[int]): Targets to treat as OOD.
            expand_3ch (bool): Whether to expand image data from 1ch to 3ch.
        """
        dataset_name = dataset_name or self.dataset_name
        emb_name = emb_name or self.emb_name
        data_dir = data_dir or self.data_dir
        emb_dir = emb_dir or self.emb_dir
        batch_size = batch_size or self.batch_size
        ood = ood or self.ood

        is_ht = self.model_name.startswith("ht_")

        # NOTE for hypothesis testing: expand_3ch=True, apply_target_transform=False
        aug_ch_3 = is_ht or expand_3ch or self.expand_3ch
        apply_target_transform = not is_ht

        if len(emb_name) > 0 and len(emb_dir) > 0:
            # embedding mode
            from datasets import get_embedding

            dm, grouping, labels = get_embedding(
                emb_dir=emb_dir,
                emb_name=emb_name,
                batch_size=batch_size,
                ood=ood,
                apply_target_transform=apply_target_transform,
            )
            assert len(dm.shape) == 1
            self.emb_dims = dm.shape[0]

        elif len(dataset_name) > 0 and len(data_dir) > 0:
            # dataset mode
            from datasets import get_data

            dm, grouping, labels = get_data(
                data_dir=data_dir,
                dataset_name=dataset_name,
                batch_size=batch_size,
                ood=ood,
                aug_ch_3=aug_ch_3,
                apply_target_transform=apply_target_transform,
            )
            assert self.emb_dims > 0

        else:
            raise ValueError()

        self.input_shape = dm.shape
        self.grouping = grouping  # NOTE must have ind_labels first and ood_labels last
        self.labels = labels  # NOTE must have ind_labels first and ood_labels last
        self.datamodule = dm

    def get_model(
        self,
        *,
        model_name: str | None = None,
        emb_dims: int | None = None,
        optim_lr: float | None = None,
        input_shape: tuple[int, ...] | None = None,
        labels: list[str] | None = None,
        cat_k: int | None = None,
    ):
        """
        Creates a model. Must be called after get_data(), as it requires
        self.input_shape, self.cat_k, and self.labels to be initialized.

        Returns:
            BaseModel: Model object

        """
        # given value or default value
        model_name = model_name or self.model_name
        emb_dims = emb_dims or self.emb_dims
        optim_lr = optim_lr or self.optim_lr
        input_shape = input_shape or self.input_shape
        labels = labels or self.labels
        cat_k = cat_k or self.cat_k

        # prerequisites
        assert input_shape is not None
        assert labels is not None
        assert cat_k is not None

        # logic
        from models import get_model

        return get_model(
            model_name=model_name,
            emb_dims=emb_dims,
            optim_lr=optim_lr,
            input_shape=input_shape,
            labels=labels,
            cat_k=cat_k,
            opt=self,
        )

    def get_transforms(
        self,
    ) -> dict:
        return {
            "input_shape": self.input_shape,
            "scale": self.scale,
            "rgb_jitter_d": self.rgb_jitter_d,
            "rgb_jitter_p": self.rgb_jitter_p,
            "rgb_gaussian_blur_p": self.rgb_gaussian_blur_p,
            "rgb_contrast": self.rgb_contrast,
            "rgb_contrast_p": self.rgb_contrast_p,
            "rgb_grid_distort_p": self.rgb_grid_distort_p,
            "rgb_grid_shuffle_p": self.rgb_grid_shuffle_p,
        }

    @property
    def cat_k(self) -> int:
        # prerequisites
        assert self.labels is not None
        assert self.ood is not None
        return len(self.labels) - len(self.ood)

    @property
    def ind_labels(self) -> List[str]:
        # prerequisites
        assert self.labels is not None
        assert self.cat_k is not None
        # logic
        return self.labels[: self.cat_k]

    @property
    def ood_labels(self) -> List[str]:
        # prerequisites
        assert self.labels is not None
        assert self.cat_k is not None
        # logic
        return self.labels[self.cat_k :]

    def print_labels(self) -> None:
        logging.info(f"Labels (train, test): {self.ind_labels}")
        logging.info(f"Labels (ood): {self.ood_labels}")

    @property
    def params(self) -> dict[str, int | str | float]:
        d = vars(self)
        d["ood"] = ":".join(map(str, self.ood))
        d.pop("dataloader", None)
        return d
