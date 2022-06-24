import torch
from einops import rearrange


def gen_patches_from_img(
  x: torch.Tensor,
  patch_h: int,
  patch_w: int,
) -> torch.Tensor:
  """
  Generate patches for a set of images

  :param x: Batch of Images
  :param patch_h: Patch Height
  :param patch_w: Patch Width
  :return:
  """
  patched = rearrange(x, 'b c (h ph) (w pw) -> b (ph pw c) h w', ph=patch_h, pw=patch_w)
  return patched


def gen_img_from_patches(
  x: torch.Tensor,
  patch_h: int,
  patch_w: int,
) -> torch.Tensor:
  """
  Generate back images from a set of image patches

  :param x: Batch of Patched Images
  :param patch_h: Patch Height
  :param patch_w: Patch Width
  :return:
  """
  unpatched = rearrange(x, 'b (ph pw c) h w -> b c (h ph) (w pw)', ph=patch_h, pw=patch_w)
  return unpatched


def set_requires_grad(
  module: torch.nn.Module,
  flag: bool
):
  """
  Sets requires_grad flag of all parameters of a torch.nn.module
  :param module: torch.nn.module
  :param flag: Flag to set requires_grad to
  """

  for param in module.parameters():
    param.requires_grad = flag


def get_best_device():
  # check for mps
  try:
    import torch.backends.mps
    mps_available = torch.backends.mps.is_available()
  except ModuleNotFoundError:
    mps_available = False

  # check for cuda
  try:
    import torch.backends.mps
    cuda_available = torch.cuda.is_available()
  except ModuleNotFoundError:
    cuda_available = False

  if cuda_available:
    return "cuda"
  elif mps_available:
    return "mps"
  return "cpu"
