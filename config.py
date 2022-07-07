import sys
from typing import Callable, Tuple

import torch.utils.data


class Config:

  def __init__(self, dataset: str):

    from dflows.nn_util import get_best_device
    from dflows.data_loaders import load_mnist, load_amrb_v1, load_amrb_v2

    # tqdm config
    self.tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}
    self.data_loader: Callable[[int, int, str], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]

    # data config
    if dataset == "MNIST":
      self.data_loader = load_mnist
      self.img_C, self.img_H, self.img_W = (1, 28, 28)
    elif dataset == "AMRB_V1":
      self.data_loader = load_amrb_v1
      self.img_C, self.img_H, self.img_W = (1, 40, 40)
    elif dataset == "AMRB_V2":
      self.data_loader = load_amrb_v2
      self.img_C, self.img_H, self.img_W = (1, 40, 40)
    else:
      raise ValueError("Only MNIST, AMRB_V1, and AMRB_V2 datasets are supported")
    self.total_dims = self.img_C * self.img_H * self.img_W

    # model config
    self.patch_H, self.patch_W = (4, 4)
    self.model_C, self.model_H, self.model_W = (self.img_C * self.patch_H * self.patch_W), (self.img_H // self.patch_H), (self.img_W // self.patch_W)
    self.manifold_dims = 1 * self.model_H * self.model_W

    # training config
    self.batch_size = 64
    self.learning_rate = 1e-4
    self.momentum = 0.5
    self.n_epochs = 100

    # paths
    self.saved_models_path = "saved_models"
    self.vis_path = "vis"
    self.data_root = "~/Documents/datasets"

    # runtime
    self.dry_run = False
    self.device = get_best_device()
