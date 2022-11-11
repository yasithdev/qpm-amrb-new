import math
from typing import Iterable, Optional

import einops
import torch

from . import Distribution


class StandardNormal(Distribution):

    """
    A multivariate Normal with zero mean and unit covariance.

    """

    def __init__(
        self,
        *shape: int,
    ) -> None:

        super().__init__()

        self._shape = torch.Size(shape)
        # TODO if needed change dtype to float64
        p = 0.5 * math.prod(shape) * math.log(2 * math.pi)
        self.register_buffer("_log_z", torch.Tensor([p]), persistent=False)

    def _log_prob(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Note: the context is ignored.
        assert self._shape == inputs.shape[1:], f"{self._shape} != {inputs.shape[1:]}"
        neg_energy = -0.5 * einops.reduce(inputs**2, "b ... -> b", reduction="sum")
        return neg_energy - self._log_z

    def _sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shape = [num_samples, *self._shape]
        device = torch.as_tensor(self._log_z).device

        # The value of the context is ignored, only its size and device are used
        if context is not None:
            shape = [context.shape[0], *shape]
            device = context.device

        samples = torch.randn(size=shape, device=device)
        return samples

    def _mean(
        self,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shape = self._shape
        device = torch.as_tensor(self._log_z).device

        # The value of the context is ignored, only its size and device are used
        if context is not None:
            shape = [context.shape[0], *shape]
            device = context.device

        return torch.zeros(size=shape, device=device)
