from typing import Optional, Tuple

import torch


class Distribution(torch.nn.Module):

    """
    Base class for all distribution objects.

    """

    def forward(
        self,
    ) -> None:
        raise RuntimeError("forward() does not apply to distributions.")

    def log_prob(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Calculate log probability under the distribution.

        Args:
            x (torch.Tensor): input tensor
            c (torch.Tensor, optional): conditioning variables (one per input)

        Returns:
            torch.Tensor: log-prob of x, or conditional log-prob if c is given

        """

        if c is not None:
            assert x.size(0) == c.size(0)

        return self._log_prob(x, c)

    def sample(
        self,
        n: int,
        c: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:

        """
        Generates samples from the distribution.
        Samples can be generated in batches.

        Args:
            n (int): number of samples to generate
            c (torch.Tensor, optional): conditioning variables
            batch_size (int, optional): batch size (default=n)

        Returns:
            torch.Tensor: generated samples
                c == None - (n, ...)
                c != None - (c_size, n, ...)

        """
        assert n > 0

        if batch_size is None:
            return self._sample(n, c)

        assert batch_size > 0
        num_batches = n // batch_size
        num_leftover = n % batch_size
        samples = [self._sample(batch_size, c) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(self._sample(num_leftover, c))
        return torch.cat(samples, dim=0)

    def mean(
        self,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Get the mean of the distribution

        Args:
            c (torch.Tensor, optional): conditioning variables

        Returns:
            torch.Tensor: mean of the distribution
        """

        return self._mean(c)

    def _log_prob(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()

    def _sample(
        self,
        n: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()

    def _mean(
        self,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()
