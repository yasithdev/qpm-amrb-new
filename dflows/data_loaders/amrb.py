import os
from typing import Tuple, Any, Optional, Callable

import numpy as np
import torch.utils.data
import torchvision

from dflows.data_loaders.util import AddGaussianNoise


class AMRB(torchvision.datasets.VisionDataset):

  def __init__(self,
               root: str,
               train: bool,
               version: int,
               transforms: Optional[Callable] = None,
               transform: Optional[Callable] = None,
               target_transform: Optional[Callable] = None,
               ) -> None:
    super().__init__(root, transforms, transform, target_transform)
    self.train = train
    self.version = version
    mode = "trn" if self.train else "tst"
    self.x_path = os.path.join(self.root, f"AMRB_V{self.version}", f"{mode}_x.npy")
    self.y_path = os.path.join(self.root, f"AMRB_V{self.version}", f"{mode}_y.npy")

    self.data_x = np.load(self.x_path)
    self.data_y = np.load(self.y_path)

    assert self.data_x.shape[0] == self.data_y.shape[0]

  def __getitem__(self, index: int) -> Any:
    img = self.data_x[index]
    target = self.data_y[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self) -> int:
    return self.data_y.shape[0]


# --------------------------------------------------------------------------------------------------------------------------------------------------


def load(
  batch_size_train: int,
  batch_size_test: int,
  data_root: str,
  version: int = 1,

) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
  transform = torchvision.transforms.Compose(
    [
      torchvision.transforms.ToTensor(),
      AddGaussianNoise(mean=0., std=.01)
    ]
  )

  # load AMRB data
  train_loader = torch.utils.data.DataLoader(
    dataset=AMRB(
      root=data_root,
      train=True,
      version=version,
      transform=transform
    ),
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    dataset=AMRB(
      root=data_root,
      train=False,
      version=version,
      transform=transform
    ),
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


# --------------------------------------------------------------------------------------------------------------------------------------------------

def load_v1(
  batch_size_train: int,
  batch_size_test: int,
  data_root: str,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
  return load(batch_size_train, batch_size_test, data_root, version=1)


# --------------------------------------------------------------------------------------------------------------------------------------------------

def load_v2(
  batch_size_train: int,
  batch_size_test: int,
  data_root: str,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
  return load(batch_size_train, batch_size_test, data_root, version=2)

# --------------------------------------------------------------------------------------------------------------------------------------------------
