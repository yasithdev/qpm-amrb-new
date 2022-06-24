from typing import Tuple

import torch.utils.data
import torchvision


# --------------------------------------------------------------------------------------------------------------------------------------------------

def load(
  batch_size_train: int,
  batch_size_test: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
  mnist_root = "~/Documents/data_loaders"
  mnist_transform = torchvision.transforms.Compose(
    [
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
  )

  # load MNIST data
  train_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST(
      root=mnist_root, train=True, download=True, transform=mnist_transform
    ),
    batch_size=batch_size_train,
    shuffle=True,
  )

  test_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST(
      root=mnist_root, train=False, download=True, transform=mnist_transform
    ),
    batch_size=batch_size_test,
    shuffle=True,
  )

  return train_loader, test_loader

# --------------------------------------------------------------------------------------------------------------------------------------------------
