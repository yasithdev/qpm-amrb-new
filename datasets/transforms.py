import torch


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
