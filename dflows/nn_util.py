import torch


# --------------------------------------------------------------------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------------------------------------------------------------------

def get_best_device():
  import torch

  # check for cuda
  if torch.cuda.is_available():
    return "cuda"

  # check for mps
  try:
    import torch.backends.mps
    if torch.backends.mps.is_available():
      return "mps"
  except ModuleNotFoundError:
    pass

  return "cpu"

# --------------------------------------------------------------------------------------------------------------------------------------------------
