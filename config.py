import os.path
import sys
from typing import Callable, Tuple

import torch.utils.data


class Config:

  def __init__(self, dataset: str):

    from dflows.nn_util import get_best_device
    from dflows.data_loaders import load_mnist, load_amrb_v1, load_amrb_v2

    # tqdm config
    self.dataset = dataset
    self.tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}
    self.data_loader: Callable[[int, int, str], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]

    # data config
    if self.dataset == "MNIST":
      self.data_loader = load_mnist
      self.img_C, self.img_H, self.img_W = (1, 28, 28)
    elif self.dataset == "AMRB_V1":
      self.data_loader = load_amrb_v1
      self.img_C, self.img_H, self.img_W = (1, 40, 40)
    elif self.dataset == "AMRB_V2":
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
    self.model_path = f"saved_models/{self.dataset}"
    self.vis_path = f"vis/{self.dataset}"
    self.data_root = os.path.expanduser("~/Documents/datasets")
    self.data_dir = os.path.join(self.data_root, self.dataset)

    os.makedirs(self.model_path, exist_ok=True)
    os.makedirs(self.vis_path, exist_ok=True)

    # runtime
    self.dry_run = False
    self.load_saved_params = False
    self.ood_mode = False
    self.device = get_best_device()

    # network configuration
    self.coupling_network = {
      'num_channels': self.model_C // 2,
      'num_hidden_channels': self.model_C,
      'num_hidden_layers': 1
    }
    self.conv1x1 = {
      'num_channels': self.model_C
    }
