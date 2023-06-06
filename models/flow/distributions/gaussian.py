from math import log, pi, prod
from typing import Optional

import einops
import torch

from . import Distribution


class StandardNormal(Distribution):

    """
    A multivariate standard normal distribution ~ N(0,I)

    """

    def __init__(
        self,
        k: int,
        mu: float = 0.0,
        std: float = 1.0,
        device: str = "cuda",
    ) -> None:

        super().__init__()

        self.k = k
        self.mu = mu
        self.std = std
        self.device = device
        # log(2π^(-k/2)) = -k/2*log(2π)
        self.const = (-self.k / 2) * log(2 * pi)

    def _log_prob(
        self,
        z: torch.Tensor,
        ignore_constants: bool = True,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        k = prod(z.size()[1:])
        assert self.k == k, f"{self.k} != {k}"
        # log(exp(-1/2*((x-mu)/std)**2))
        dist = (z - self.mu) / self.std
        log_exp = (-1 / 2) * einops.reduce(dist**2, "b ... -> b", reduction="sum")
        # -k/2*log(2π) + log(exp(-1/2*((x-mu)/std)**2))
        if ignore_constants:
            return log_exp
        else:
            return self.const + log_exp

    def _sample(
        self,
        n: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        sample = torch.randn(size=(n, self.k), device=self.device)
        return sample

    def _mean(
        self,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        mean = torch.zeros(self.k, device=self.device)
        return mean
