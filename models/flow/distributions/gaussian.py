import math
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
        self.log_z = 0.5 * math.prod(shape) * math.log(2 * math.pi)

    def _log_prob(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: c is ignored
        assert self.shape == x.shape[1:], f"{self.shape} != {x.shape[1:]}"
        neg_energy = -0.5 * einops.reduce(x**2, "b ... -> b", reduction="sum")
        return neg_energy - self.log_z

    def _sample(
        self,
        n: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shape = [n, *self.shape]
        device = torch.as_tensor(self.log_z).device

        # The value of the c is ignored, only its size and device are used
        if c is not None:
            shape = [c.shape[0], *shape]
            device = c.device

        samples = torch.randn(size=shape, device=device)
        return samples

    def _mean(
        self,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shape = self.shape
        device = torch.as_tensor(self.log_z).device

        # The value of the c is ignored, only its size and device are used
        if c is not None:
            shape = [c.shape[0], *shape]
            device = c.device

        return torch.zeros(size=shape, device=device)
