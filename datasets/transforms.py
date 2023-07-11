from typing import List

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


class AddGaussianNoise(object):
    def __init__(self, mean: float, std: float):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ZeroPad2D(object):
    def __init__(self, h1: int, h2: int, w1: int, w2: int):
        self.h1 = h1
        self.h2 = h2
        self.w1 = w1
        self.w2 = w2

    def __call__(self, tensor: torch.Tensor):
        pad = (self.w1, self.w2, self.h1, self.h2)
        return torch.constant_pad_nd(tensor, pad)

    def __repr__(self):
        return self.__class__.__name__ + "(H={0},{1}, W={2}{3})".format(
            self.h1, self.h2, self.w1, self.w2
        )


class TileChannels2d(object):
    def __init__(self, repeats: int):
        self.repeats = repeats

    def __call__(self, tensor: torch.Tensor):
        return tensor.tile((self.repeats, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + "(repeats={0})".format(self.repeats)


def create_target_transform(
    targets: List[str],
    ood: List[int],
) -> Compose:
    K = len(targets) - len(ood)
    ind_targets = [x for i, x in enumerate(targets) if i not in ood]
    ood_targets = [x for i, x in enumerate(targets) if i in ood]
    permuted_targets = ind_targets + ood_targets
    target_transform = Compose(
        [
            lambda x: permuted_targets.index(x),
            lambda x: torch.tensor(x),
            lambda x: F.one_hot(x, K) if x < K else torch.zeros(K),
        ]
    )
    return target_transform
