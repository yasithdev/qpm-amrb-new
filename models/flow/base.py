from typing import Optional, Tuple, List

import torch


class FlowTransform(torch.nn.Module):
    """
    Base Class for Flow Transforms.
    It provides _forward() and _inverse() directives to implement bijections
    """

    def forward(
        self,
        data: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        forward: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if forward:
            return self._forward(data, c)
        else:
            return self._inverse(data, c)

    def _forward(
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

    def _inverse(
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


class Compose(FlowTransform):
    """
    Composition of Flow Transformations.
    Each transformation operates on all the dims.

    Example:

    A-B-C-D-E-> \n
    A-B-C-D-E-> \n
    A-B-C-D-E-> \n
    A-B-C-D-E-> \n
    A-B-C-D-E-> \n
    A-B-C-D-E-> \n
    A-B-C-D-E-> \n
    A-B-C-D-E-> \n

    """

    def __init__(
        self,
        transforms: List[FlowTransform],
    ) -> None:

        super().__init__()

        self.transforms = torch.nn.ModuleList(transforms)

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)

        # initialize h=x
        h = x

        # total logabsdet (initially 0)
        sum_logabsdet = x.new_zeros(B)

        # iterate in order
        for transform in self.transforms:
            assert isinstance(transform, FlowTransform)

            # transform h and accumulate log_det
            h, logabsdet = transform(h, forward=True)
            sum_logabsdet = sum_logabsdet + logabsdet

        z = h
        return z, sum_logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = z.size(0)

        # initialize h=x
        h = z

        # total logabsdet (initially 0)
        sum_logabsdet = z.new_zeros(B)

        # iterate in reverse
        for transform in reversed(self.transforms):
            assert isinstance(transform, FlowTransform)

            # transform h and accumulate log_det
            h, logabsdet = transform(h, forward=False)
            sum_logabsdet = sum_logabsdet + logabsdet
        x = h
        return x, sum_logabsdet


class ComposeMultiScale(FlowTransform):
    """
    Composition of Multi-Scale Flow Transformations.
    Each transformation operates on half the previous dims.

    Example:

    A-B-------> \n
    A-B-------> \n
    A-B-------> \n
    A-B-------> \n
    A-B-C-----> \n
    A-B-C-----> \n
    A-B-C-D---> \n
    A-B-C-D-E-> \n

    """

    def __init__(
        self,
        transforms: List[FlowTransform],
    ) -> None:

        super().__init__()

        self.transforms = torch.nn.ModuleList(transforms)
        self.dim = 1

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # initialize h=x
        h = x

        # initialize det=1 => log_det=0
        sum_logabsdet = torch.zeros(h.size(0), device=x.device)

        # define initial slice lengths
        total_length = h.size(self.dim)
        slice_length = total_length

        # iterate in order
        for transform in self.transforms:
            assert isinstance(transform, FlowTransform)

            # split h for multi-scale transformation
            h1 = torch.narrow(h, self.dim, 0, slice_length)
            h2 = torch.narrow(h, self.dim, slice_length, total_length - slice_length)

            # transform h and accumulate log_det
            h1, logabsdet = transform(h1, forward=True)
            h = torch.cat([h1, h2], self.dim)

            # halve the slice length for next transform
            slice_length //= 2

            # accumulate logabsdet
            sum_logabsdet = sum_logabsdet + logabsdet

        z = h
        return z, sum_logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # initialize h=z
        h = z

        # initialize det=1 => log_det=0
        sum_logabsdet = torch.zeros(h.size(0), device=z.device)

        # define initial slice lengths
        total_length = h.size(self.dim)
        slice_length = total_length // 2 ** (len(self.transforms))

        # iterate in reverse
        for transform in reversed(self.transforms):
            assert isinstance(transform, FlowTransform)

            # split h for multi-scale transformation
            h1 = torch.narrow(h, self.dim, 0, slice_length)
            h2 = torch.narrow(h, self.dim, slice_length, total_length - slice_length)

            # transform h and accumulate log_det
            h1, logabsdet = transform(h1, forward=False)
            h = torch.cat([h1, h2], self.dim)

            # halve the slice length for next transform
            slice_length *= 2

            # accumulate logabsdet
            sum_logabsdet = sum_logabsdet + logabsdet

        x = h
        return x, sum_logabsdet
