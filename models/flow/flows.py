from typing import Optional

import torch

from . import Distribution, FlowTransform


class Flow(Distribution):

    """
    Base class for all flows (Bijective, Injective, Surjective)

    """

    def __init__(
        self,
        transform: FlowTransform,
        base_dist: Distribution,
    ) -> None:

        """
        Create a flow with a transform to/from a base distribution

        Args:
            base_dist (Distribution): base distribution
            transform (FlowTransform): transformation to/from the base distribution
        """

        super().__init__()

        self.transform = transform
        self.base_dist = base_dist

    def _log_prob(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Using the transform (F) to map the input (x) into a latent (z), then compute log p(x) using p(z).

        Unconditional:  log p(x)   = log p(z)   + log |∇[F](z)|
        Conditional:    log p(x|c) = log p(z|c) + log |∇[F](z,c)|

        Args:
            x (torch.Tensor): input (x)
            c (Optional[torch.Tensor], optional): optional context vector

        Returns:
            torch.Tensor: log p(x)

        """

        z, logabsdet = self.transform(x, forward=True)
        log_p_z = self.base_dist.log_prob(z, c)
        return log_p_z + logabsdet

    def _sample(
        self,
        n: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Args:
            n (int): number of samples
            c (Optional[torch.Tensor], optional): conditioning variables

        Returns:
            torch.Tensor: samples from p(x)

        """

        z = self.base_dist.sample(n, c)
        x, _ = self.transform(z, c, forward=False)
        return x
