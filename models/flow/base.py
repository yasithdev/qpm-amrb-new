from typing import Optional, Tuple

import torch


class FlowTransform(torch.nn.Module):
    """
    Base Class for Flow Transforms.
    It provides forward() and inverse() directives to implement bijections
    """

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        transform a tensor towards latent space (forward mode)

        :param x: Tensor
        :return: Tuple [Transformed Tensor, Log Absolute Determinant of Jacobian]

        """
        raise NotImplementedError()

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        transform a tensor towards data space (inverse mode)

        :param z: Tensor
        :return: Tuple [Transformed Tensor, Log Absolute Determinant of Jacobian]
        """
        raise NotImplementedError()
