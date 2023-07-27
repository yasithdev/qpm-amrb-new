import einops
import numpy as np
import torch
from tqdm import tqdm

import models.flow as flow
from config import Config, load_config
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
from models import get_model_optimizer_and_step
from models.common import load_model_state


def main(config: Config):

    assert config.train_loader
    assert config.test_loader

    model, optim, _ = get_model_optimizer_and_step(config)

    # load saved model and optimizer
    load_model_state(
        model=model,
        config=config,
        epoch=config.train_epochs,
    )
    model = model.float().to(config.device)

    # run tests
    flow_x: flow.Compose = model["flow_x"]  # type: ignore
    flow_u: flow.Compose = model["flow_u"]  # type: ignore

    x: torch.Tensor
    y: torch.Tensor

    uv_x: torch.Tensor
    u_x: torch.Tensor
    v_x: torch.Tensor
    z_x: torch.Tensor

    u_z: torch.Tensor
    v_z: torch.Tensor
    x_z: torch.Tensor

    # get one batch of data
    x0: torch.Tensor
    x0, _ = next(config.train_loader.__iter__())
    x0 = x0.float().to(config.device)
    B = x0.size(0)

    # generate plot
    vars = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for var in tqdm(vars, desc="Noise Levels", **config.tqdm_args):

        # cast x and y to float
        e = torch.randn(x0.size(), device=x0.device) * var
        x = x0 + e

        # x -> (u x v) -> u -> z
        uv_x, _ = flow_x(x, forward=True)
        u_x, v_x = flow.nn.partition(uv_x, config.manifold_d)
        z_x, _ = flow_u(u_x, forward=True)
        uX = (v_x).pow(2).flatten(start_dim=1).sum(dim=-1)

        # u -> (u x 0) -> x~
        u_z = u_x
        v_z = v_x.new_zeros(v_x.size())
        uv_z = flow.nn.join(u_z, v_z)
        x_z, _ = flow_x(uv_z, forward=False)

        HWC = lambda x: einops.rearrange(x, "B C H W -> B H W C").cpu().numpy()
        SQ_SUM = lambda x: x.flatten(start_dim=1).pow(2).sum(dim=-1).cpu().numpy()
        JOIN = lambda ab: np.concatenate(ab, axis=1)

        dX: np.ndarray = JOIN([HWC(x.detach()), HWC(x_z.detach())])
        dV: np.ndarray = SQ_SUM(v_x.detach())

        row = [wandb.Image(dX[i], caption=f"{dV[i]:.4f}") for i in range(10)]
        wandb.log({f"comparison/N={var}": row})


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)

    config = load_config()

    # get dataset info
    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        data_root=config.data_dir,
        cv_mode=config.cv_mode,
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
    )
    config.print_labels()

    import wandb.plot

    import wandb

    wandb.init(
        project="ood_flows",
        name=config.run_name,
        config=config.run_config,
    )

    main(config)
