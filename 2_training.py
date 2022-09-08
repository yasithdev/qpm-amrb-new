import logging
import os
import shutil

import numpy as np
import torch

from config import Config, load_config
from models import get_model_optimizer_and_loops


def main(config: Config):

    # set experiment name and path
    experiment_name = "2_training"
    experiment_path = os.path.join(
        config.experiment_dir,
        experiment_name,
        config.dataset_name,
        str(config.crossval_k),
    )
    if not (config.exc_resume or config.exc_dry_run):
        shutil.rmtree(experiment_path, ignore_errors=True)
        os.makedirs(experiment_path, exist_ok=True)

    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(config.manifold_c), torch.eye(config.manifold_c)
    )

    model, optim, train_model, test_model = get_model_optimizer_and_loops(
        name="nf",
        config=config,
        experiment_path=experiment_path,
    )

    train_stats = []
    test_stats = []

    # run train / test loops
    logging.info("Started Train/Test")
    for current_epoch in range(config.train_epochs + 1):
        # training loop
        if current_epoch > 0:
            train_model(
                nn=model,
                epoch=current_epoch,
                config=config,
                optim=optim,
                stats=train_stats,
                experiment_path=experiment_path,
            )
        # testing loop
        test_model(
            nn=model,
            epoch=current_epoch,
            config=config,
            stats=test_stats,
            experiment_path=experiment_path,
        )


if __name__ == "__main__":
    # initialize the RNG deterministically
    np.random.seed(42)
    torch.random.manual_seed(42)
    config = load_config()
    main(config)
