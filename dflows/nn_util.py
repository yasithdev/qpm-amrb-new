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

# --------------------------------------------------------------------------------------------------------------------------------------------------
