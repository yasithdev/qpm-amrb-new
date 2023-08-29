import logging
import os
from typing import Tuple, List
from dotenv import load_dotenv
from lightning.pytorch import LightningDataModule

import argparse

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
        checkpoint_mode: str,
        args : argparse.Namespace
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
        self.checkpoint_mode = checkpoint_mode
        self.image_chw: Tuple[int, int, int] | None = None
        self.labels: List[str] | None = None
        self.datamodule: LightningDataModule | None = None
        self.ood: List[int] = []
        self.args = args
        for k in self.ood_k.split(':'):
            if len(k) > 0:
                self.ood.append(int(k))
        # model-dependent data params
        if model_name.startswith("resnet18") or model_name.startswith("resnet50"):
            self.args.expand_3ch = True
        else:
            self.args.expand_3ch = False

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
            aug_ch_3 = self.args.expand_3ch,
        )
        self.datamodule = dm
        self.labels = dm.targets
        self.image_chw = dm.shape
        self.args.image_size = dm.shape # NOTE this should be set for simclr_transform() to work

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
            self.optim_lr,
            self.args,
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
            checkpoint_metric=self.checkpoint_metric,
            checkpoint_mode=self.checkpoint_mode,
        )

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
    parser.add_argument('--ood_k', **default_env("OOD_K"))
    parser.add_argument('--data_dir', **default_env("DATA_DIR"))
    parser.add_argument('--dataset_name', **default_env("DATASET_NAME"))
    parser.add_argument('--model_name', **default_env("MODEL_NAME"))
    parser.add_argument('--experiment_base', **default_env("EXPERIMENT_BASE"))
    parser.add_argument('--manifold_d', **default_env("MANIFOLD_D"))
    parser.add_argument('--batch_size', **default_env("BATCH_SIZE"))
    parser.add_argument('--optim_lr', **default_env("OPTIM_LR"))
    parser.add_argument('--optim_m', **default_env("OPTIM_M"))
    parser.add_argument('--train_epochs', **default_env("TRAIN_EPOCHS"))
    parser.add_argument('--checkpoint_metric', **default_env("CHECKPOINT_METRIC"))
    parser.add_argument('--checkpoint_mode', **default_env("CHECKPOINT_MODE"))
    
    ## SSL Augmentation parameters
    # Common augmentations
    parser.add_argument("--scale",  nargs=2, type=float, default=[0.2, 1.0])
    parser.add_argument("--train_supervised", type=bool, default=False)
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')

    # RGB augmentations
    parser.add_argument("--rgb_gaussian_blur_p", type=float, default=0, help="probability of using gaussian blur (only on rgb)" )
    parser.add_argument("--rgb_jitter_d", type=float, default=1, help="color jitter 0.8*d, val 0.2*d (only on rgb)" )
    parser.add_argument("--rgb_jitter_p", type=float, default=0.8, help="probability of using color jitter(only on rgb)" )
    parser.add_argument("--rgb_contrast", type=float, default=0.2, help="value of contrast (rgb only)")
    parser.add_argument("--rgb_contrast_p", type=float, default=0, help="prob of using contrast (rgb only)")
    parser.add_argument("--rgb_grid_distort_p", type=float, default=0, help="probability of using grid distort (only on rgb)" )
    parser.add_argument("--rgb_grid_shuffle_p", type=float, default=0, help="probability of using grid shuffle (only on rgb)" )

    args, _ = parser.parse_known_args()
    logging.info(args.__dict__)

    return Config(
        ood_k=str(args.ood_k),
        data_dir=str(args.data_dir),
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        experiment_base=str(args.experiment_base),
        manifold_d=int(args.manifold_d),
        batch_size=int(args.batch_size),
        optim_lr=float(args.optim_lr),
        optim_m=float(args.optim_m),
        train_epochs=int(args.train_epochs),
        checkpoint_metric=str(args.checkpoint_metric),
        checkpoint_mode=str(args.checkpoint_mode),
        args = args
    )
