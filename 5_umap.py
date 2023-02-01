import os

import numpy as np
import pandas as pd
import torch

from config import Config, load_config
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
from models import get_model_optimizer_and_loops
from models.common import load_saved_state


def main(
    df: pd.DataFrame,
    config: Config,
) -> None:

    model, optim, train_model, test_model = get_model_optimizer_and_loops(config)

    rows = df[
        (df.model == config.model_name) & (df.cv_k == config.cv_k)
    ]
    epochs = list(rows.step)
    print(f"Epochs: {epochs}")

    # for each selected epoch, load saved model and optimizer parameters
    for epoch in epochs:
        # load_saved_state(
        #     model=model,
        #     optim=optim,
        #     experiment_path=experiment_path,
        #     config=config,
        #     epoch=epoch,
        # )
        # model = model.float().to(config.device)

        # train_labels = list(config.train_loader.dataset.labels)
        # test_labels = list(config.test_loader.dataset.labels)
        # labels = (
        #     [*train_labels, *test_labels] if config.cv_mode == "leave-out" else train_labels
        # )

        input(f"Epoch {epoch}: DONE")

        # # testing loop
        # stats = test_model(
        #     model=model,
        #     epoch=epoch,
        #     config=config,
        # )


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(mode=True)

    config = load_config()

    # get dataset info
    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        data_root=config.data_dir,
        cv_mode=config.cv_mode,
        label_type=config.label_type,
    )
    # get image dims
    config.image_chw = get_dataset_chw(
        dataset_name=config.dataset_name,
    )
    # initialize data loaders
    config.train_loader, config.test_loader = get_dataset_loaders(
        dataset_name=config.dataset_name,
        batch_size_train=config.batch_size,
        batch_size_test=config.batch_size,
        data_root=config.data_dir,
        cv_k=config.cv_k,
        cv_folds=config.cv_folds,
        cv_mode=config.cv_mode,
        label_type=config.label_type,
    )

    experiment_path = os.path.join(
        config.experiment_dir,
        "2_training",
        f"{config.dataset_name}-{config.model_name}",
        f"{config.label_type}-{config.cv_k}",
    )

    df_path = f"results/{config.dataset_name}_cls_summary_{config.cv_mode}_{config.label_type}.csv"
    df = pd.read_csv(df_path)

    main(df, config)
