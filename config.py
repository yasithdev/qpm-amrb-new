import argparse
import logging
import os
from typing import List, Tuple

from dotenv import load_dotenv
from lightning.pytorch import LightningDataModule


class Config(argparse.Namespace):
    model_name: str
    ood: list[int]
    data_dir: str
    dataset_name: str
    experiment_base: str
    manifold_d: int
    batch_size: int
    optim_lr: float
    optim_m: float
    train_epochs: int
    checkpoint_metric: str
    checkpoint_mode: str
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
    image_chw: Tuple[int, int, int] | None = None
    labels: List[str] | None = None
    cat_k: int | None = None
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
            aug_ch_3=self.expand_3ch,
        )
        self.image_chw = dm.shape
        self.image_size = dm.shape  # NOTE this should be set for simclr_transform() to work
        self.labels = dm.permuted_targets  # NOTE must have ind_labels first and ood_labels last
        self.cat_k = len(self.labels) - len(self.ood)
        self.datamodule = dm

    def load_embedding(
        self,
        *,
        embedding_path: str,
        num_dims: int,
        num_targets: int,
        batch_size: int | None = None,
        ood: list[int] | None = None,
    ) -> None:
        from datasets import get_embedding

        dm = get_embedding(
            embedding_path=embedding_path,
            num_dims=num_dims or self.manifold_d,
            num_targets=num_targets,
            batch_size=batch_size or self.batch_size,
        )
        self.image_chw = (num_dims, 1, 1)
        # self.image_size = (num_dims, 1, 1)  # NOTE this should be set for simclr_transform() to work
        # self.labels = dm.permuted_targets  # NOTE must have ind_labels first and ood_labels last
        self.labels = [str(x) for x in range(num_targets)]
        self.cat_k = len(self.labels) - len(self.ood)
        self.datamodule = dm

    def get_model(
        self,
        *,
        model_name: str | None = None,
        manifold_d: int | None = None,
        optim_lr: float | None = None,
        image_chw: tuple[int, int, int] | None = None,
        labels: list[str] | None = None,
        cat_k: int | None = None,
        **kwargs,
    ):
        """
        Creates a model. Must be called after get_data(), as it requires
        self.image_chw, self.cat_k, and self.labels to be initialized.

        Returns:
            BaseModel: Model object

        """
        # given value or default value
        image_chw = image_chw or self.image_chw
        labels = labels or self.labels
        cat_k = cat_k or self.cat_k

        # prerequisites
        assert image_chw is not None
        assert labels is not None
        assert cat_k is not None

        # logic
        from models import get_model

        return get_model(
            model_name=model_name or self.model_name,
            manifold_d=manifold_d or self.manifold_d,
            optim_lr=optim_lr or self.optim_lr,
            image_chw=image_chw,
            labels=labels,
            cat_k=cat_k,
            opt=self,
            **kwargs,
        )

    def get_ind_labels(self) -> List[str]:
        # prerequisites
        assert self.labels is not None
        assert self.cat_k is not None
        # logic
        return self.labels[: self.cat_k]

    def get_ood_labels(self) -> List[str]:
        # prerequisites
        assert self.labels is not None
        assert self.cat_k is not None
        # logic
        return self.labels[self.cat_k :]

    def print_labels(
        self,
    ) -> None:
        logging.info(f"Labels (train, test): {self.get_ind_labels()}")
        logging.info(f"Labels (ood): {self.get_ood_labels()}")

    def as_dict(
        self,
    ) -> dict[str, int | str | float]:
        d = vars(self)
        d["ood"] = ":".join(map(str, self.ood))
        d.pop("dataloader", None)
        return d


def default_env(key) -> dict:
    val = os.environ.get(key, default="")
    val = os.path.expanduser(val)
    val = os.path.expandvars(val)
    return dict(default=val)


def load_config() -> Config:
    log_level = logging.WARN
    logging.basicConfig(level=log_level)
    logging.info(f"LOG_LEVEL={log_level}")

    import matplotlib

    matplotlib.rcParams["figure.figsize"] = [16, 12]

    load_dotenv()

    parser = argparse.ArgumentParser(description="configuration")
    parser.add_argument("--data_dir", type=str, **default_env("DATA_DIR"))
    parser.add_argument("--dataset_name", type=str, **default_env("DATASET_NAME"))
    parser.add_argument("--model_name", type=str, **default_env("MODEL_NAME"))
    parser.add_argument("--experiment_base", type=str, **default_env("EXPERIMENT_BASE"))
    parser.add_argument("--manifold_d", type=int, **default_env("MANIFOLD_D"))
    parser.add_argument("--batch_size", type=int, **default_env("BATCH_SIZE"))
    parser.add_argument("--optim_lr", type=float, **default_env("OPTIM_LR"))
    parser.add_argument("--optim_m", type=float, **default_env("OPTIM_M"))
    parser.add_argument("--train_epochs", type=int, **default_env("TRAIN_EPOCHS"))
    parser.add_argument("--checkpoint_metric", type=str, **default_env("CHECKPOINT_METRIC"))
    parser.add_argument("--checkpoint_mode", type=str, **default_env("CHECKPOINT_MODE"))
    parser.add_argument("--ood", type=str, **default_env("OOD"))

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

    args, _ = parser.parse_known_args(namespace=Config())
    return args
