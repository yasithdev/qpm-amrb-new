import logging
import os
import shutil

import numpy as np
import torch

from config import Config, load_config
from models import get_model_optimizer_and_loops
from models.common import load_saved_state, save_state


def stats_to_wandb_log(
    stats: dict,
    labels: list,
    prefix: str,
) -> dict:
    return {
        f"{prefix}_loss": stats["loss"],
        f"{prefix}_accuracy": stats["acc"],
        f"{prefix}_targets": [
            wandb.Image(x, caption=labels[y]) for x, y, _, _ in stats["samples"]
        ],
        f"{prefix}_estimates": [
            wandb.Image(x_z, caption=labels[y_z]) for _, _, x_z, y_z in stats["samples"]
        ],
        f"{prefix}_cm": wandb.plot.confusion_matrix(
            y_true=stats["y_true"],
            preds=stats["y_pred"],
            class_names=labels,
            title=f"Confusion Matrix ({prefix})",
        ),
    }


def main(config: Config):

    if not (config.exc_resume or config.exc_dry_run):
        shutil.rmtree(experiment_path, ignore_errors=True)
        os.makedirs(experiment_path, exist_ok=True)

    z_dist = torch.distributions.MultivariateNormal(
        torch.zeros(config.manifold_c), torch.eye(config.manifold_c)
    )

    model, optim, train_model, test_model = get_model_optimizer_and_loops(config)

    # load saved model and optimizer, if present
    if config.exc_resume:
        load_saved_state(
            model=model,
            optim=optim,
            experiment_path=experiment_path,
            config=config,
        )
    model = model.float().to(config.device)

    wandb.watch(model, log_freq=100)

    # run train / test loops
    logging.info("Started Train/Test")
    labels = config.train_loader.dataset.labels

    for epoch in range(config.train_epochs + 1):

        # training loop
        if epoch > 0:
            stats = train_model(
                model=model,
                epoch=epoch,
                config=config,
                optim=optim,
                z_dist=z_dist,
            )
            wandb.log(stats_to_wandb_log(stats, labels, prefix="train"), step=epoch)

        # testing loop
        stats = test_model(
            model=model,
            epoch=epoch,
            config=config,
            z_dist=z_dist,
        )
        wandb.log(stats_to_wandb_log(stats, labels, prefix="test"), step=epoch)

        # save model/optimizer states
        if not config.exc_dry_run:
            save_state(
                model=model,
                optim=optim,
                experiment_path=experiment_path,
                epoch=epoch,
            )

        model_savename = f"{config.model_name}-{config.label_type}-{config.cv_k}"
        artifact = wandb.Artifact(model_savename, type="model")
        artifact.add_file(os.path.join(experiment_path, f"model_e{epoch}.pth"))
        artifact.add_file(os.path.join(experiment_path, f"optim_e{epoch}.pth"))
        wandb.log_artifact(artifact)


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)
    # torch.use_deterministic_algorithms(mode=True)

    config = load_config()

    # set experiment name and path
    experiment_name = "2_training"
    experiment_path = os.path.join(
        config.experiment_dir,
        experiment_name,
        f"{config.dataset_name}-{config.model_name}",
        f"{config.label_type}-{config.cv_k}",
    )
    name_tags = [config.dataset_name, config.model_name, config.cv_mode, str(config.cv_k)]
    if config.dataset_name.startswith("AMRB"):
        name_tags.insert(1, config.label_type)

    import wandb

    wandb.init(
        project="qpm-amrb",
        name="-".join(name_tags),
        config={
            "cv_folds": config.cv_folds,
            "cv_k": config.cv_k,
            "cv_mode": config.cv_mode,
            "dataset": config.dataset_name,
            "model": config.model_name,
        },
    )

    main(config)
