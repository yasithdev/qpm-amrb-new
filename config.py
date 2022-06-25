import sys


class Config:
  # tqdm config
  tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}

  # data config
  img_C, img_H, img_W = (1, 28, 28)
  total_dims = img_C * img_H * img_W

  # model config
  patch_H, patch_W = (4, 4)
  model_C, model_H, model_W = (img_C * patch_H * patch_W), (img_H // patch_H), (img_W // patch_W)
  manifold_dims = 1 * model_H * model_W

  # training config
  batch_size = 32
  learning_rate = 1e-4
  momentum = 0.2
  n_epochs = 100

  # paths
  saved_models_path = "saved_models"
  vis_path = "vis"
  data_root = "~/Documents/datasets"

  # runtime
  dry_run = False
