import os

import torch

import dflows as nf
import dflows.transforms as nft
from config import load_config
from util.loops import test_model, train_model

if __name__ == "__main__":

    # set up config
    config = load_config()
    print(f"Using device: {config.device}")

    train_loader, test_loader = config.data_loader()

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
        model_state_path = os.path.join(
            config.saved_model_dir, config.dataset_name, config.crossval_k, "model.pth"
        )
        optim_state_path = os.path.join(
            config.saved_model_dir, config.dataset_name, config.crossval_k, "optim.pth"
        )

        if os.path.exists(model_state_path):
            ambient_model.load_state_dict(
                torch.load(model_state_path, map_location=config.device)
            )
            print("Loaded saved model state from:", model_state_path)

        if os.path.exists(optim_state_path):
            optim.load_state_dict(
                torch.load(optim_state_path, map_location=config.device)
            )
            print("Loaded saved optim state from:", optim_state_path)

    # run train / test loops
    print("\nStarted Train/Test")
    test_model(
        nn=ambient_model,
        epoch=0,
        loader=test_loader,
        config=config,
        test_losses=test_losses,
    )
    for current_epoch in range(1, config.train_epochs + 1):
        train_model(
            nn=ambient_model,
            epoch=current_epoch,
            loader=train_loader,
            config=config,
            optim=optim,
            train_losses=train_losses,
        )
        test_model(
            nn=ambient_model,
            epoch=current_epoch,
            loader=test_loader,
            config=config,
            test_losses=test_losses,
        )
