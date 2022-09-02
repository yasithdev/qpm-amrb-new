import logging
import os
import shutil

import numpy as np
import torch

import dflows as nf
import dflows.transforms as nft
from config import Config, load_config
from loops import test_model, train_model


def main(config: Config):

    train_loader, test_loader = config.data_loader()

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

    # create flow (pending)
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(config.manifold_c), torch.eye(config.manifold_c)
    )

    ambient_model = nf.SquareNormalizingFlow(
        transforms=[
            nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network_config)),
            nft.Conv1x1(**config.conv1x1_config),
            nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network_config)),
            nft.Conv1x1(**config.conv1x1_config),
            nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network_config)),
            nft.Conv1x1(**config.conv1x1_config),
            nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network_config)),
            nft.Conv1x1(**config.conv1x1_config),
        ]
    )

    # set up optimizer
    optim_config = {"params": ambient_model.parameters(), "lr": config.optim_lr}
    optim = torch.optim.Adam(**optim_config)

    # list to store train / test losses
    train_losses = []
    test_losses = []

    # load saved model and optimizer, if present
    if config.exc_resume:
        model_state_path = os.path.join(experiment_path, "model.pth")
        optim_state_path = os.path.join(experiment_path, "optim.pth")

        if os.path.exists(model_state_path):
            ambient_model.load_state_dict(
                torch.load(model_state_path, map_location=config.device)
            )
            logging.info("Loaded saved model state from:", model_state_path)

        if os.path.exists(optim_state_path):
            optim.load_state_dict(
                torch.load(optim_state_path, map_location=config.device)
            )
            logging.info("Loaded saved optim state from:", optim_state_path)

    # run train / test loops
    logging.info("Started Train/Test")
    test_model(
        nn=ambient_model,
        epoch=0,
        loader=test_loader,
        config=config,
        test_losses=test_losses,
        experiment_path=experiment_path,
    )
    for current_epoch in range(1, config.train_epochs + 1):
        train_model(
            nn=ambient_model,
            epoch=current_epoch,
            loader=train_loader,
            config=config,
            optim=optim,
            train_losses=train_losses,
            experiment_path=experiment_path,
        )
        test_model(
            nn=ambient_model,
            epoch=current_epoch,
            loader=test_loader,
            config=config,
            test_losses=test_losses,
            experiment_path=experiment_path,
        )


if __name__ == "__main__":
    # initialize the RNG deterministically
    np.random.seed(42)
    torch.random.manual_seed(42)
    config = load_config()
    main(config)
