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
        *shape: int,
    ) -> None:

        super().__init__()

        self.shape = torch.Size(shape)
        k = prod(shape)
        # -1/2.k.log(2π)
        self.neg_log_pz = torch.as_tensor(-0.5 * k * log(2 * pi))

    def _log_prob(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        assert self.shape == z.shape[1:], f"{self.shape} != {z.shape[1:]}"
        # -1/2.log(xTx)
        log_exp = -0.5 * einops.reduce(z**2, "b ... -> b", reduction="sum")
        # -1/2.k.log(2π) + -1/2.log(xTx)
        log_prob = self.neg_log_pz + log_exp
        return log_prob

    def _sample(
        self,
        n: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        shape = [n, *self.shape]
        device = self.neg_log_pz.device
        sample = torch.randn(size=shape, device=device)
        return sample

    def _mean(
        self,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        mean = self.neg_log_pz.new_zeros(self.shape)
        return mean
