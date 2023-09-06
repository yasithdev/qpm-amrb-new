from typing import Optional, Tuple

import einops
import torch
import torch.nn.functional as F
import math

from . import FlowTransform


class Flatten(FlowTransform):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
    ) -> None:

        super().__init__()
        self.input_shape = input_shape

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        return x.reshape(B, math.prod(self.input_shape)), x.new_zeros(B)

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = z.size(0)
        return z.reshape(B, *self.input_shape), z.new_zeros(B)


class Unflatten(FlowTransform):
    def __init__(
        self,
        target_shape: Tuple[int, ...],
    ) -> None:

        super().__init__()
        self.target_shape = target_shape

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        return x.reshape(B, *self.target_shape), x.new_zeros(B)

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = z.size(0)
        return z.reshape(B, math.prod(self.target_shape)), z.new_zeros(B)


class Pad(FlowTransform):
    def __init__(
        self,
        padding: int,
    ) -> None:

        super().__init__()
        self.padding = padding

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        pattern = [0, 0, 0, 0, 0, self.padding]
        return F.pad(x, pattern), x.new_zeros(B)

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C = z.size(0), z.size(1)
        return z.narrow(dim=1, start=0, length=C - self.padding), z.new_zeros(B)


class Proj(FlowTransform):
    def __init__(
        self,
        ambient_dims: int,
        emb_dimsims: int,
    ) -> None:

        super().__init__()
        self.ambient_dims = ambient_dims
        self.emb_dimsims = emb_dimsims
        self.padding = self.ambient_dims - self.emb_dimsims

    def _inverse(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        pattern = [0, 0, 0, 0, 0, self.padding]
        return F.pad(x, pattern), x.new_zeros(B)

    def _forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C = z.size(0), z.size(1)
        return z.narrow(dim=1, start=0, length=self.emb_dimsims), z.new_zeros(B)


class Squeeze(FlowTransform):

    """
    An image transform that squeezes spatial dims into channel dims by a given factor

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    """

    def __init__(
        self,
        factor: int = 2,
    ) -> None:

        super().__init__()

        assert factor > 1, "Factor must be an integer > 1."
        self.factor = factor

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() == 4, "Expecting inputs with 4 dimensions"

        B, _, H, W = x.size()
        F = self.factor

        assert H % F == 0, f"H ({H}) not divisible by factor ({F})"
        assert W % F == 0, f"W ({W}) not divisible by factor ({F})"

        pattern = "B C (H fH) (W fW) -> B (C fH fW) H W"
        z = einops.rearrange(x, pattern, fH=F, fW=F)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() == 4, "Expecting inputs with 4 dimensions"

        B, C, _, _ = z.size()
        F = self.factor

        assert C % (F**2) == 0, f"C ({C}) not divisible by factor^2 ({F**2})"

        pattern = "B (C fH fW) H W -> B C (H fH) (W fW)"
        x = einops.rearrange(z, pattern, fH=F, fW=F)
        logabsdet = z.new_zeros(B)

        return x, logabsdet


class Unsqueeze(FlowTransform):

    """
    An image transform that unsqueezes spatial dims into channel dims by a given factor

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    """

    def __init__(
        self,
        factor: int = 2,
    ) -> None:

        super().__init__()

        assert factor > 1, "Factor must be an integer > 1."
        self.factor = factor

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.dim() == 4, "Expecting inputs with 4 dimensions"

        B, C, _, _ = x.size()
        F = self.factor

        assert C % (F**2) == 0, f"C ({C}) not divisible by factor^2 ({F**2})"

        pattern = "B (C fH fW) H W -> B C (H fH) (W fW)"
        z = einops.rearrange(x, pattern, fH=F, fW=F)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def _inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert z.dim() == 4, "Expecting inputs with 4 dimensions"

        B, _, H, W = z.size()
        F = self.factor

        assert H % F == 0, f"H ({H}) not divisible by factor ({F})"
        assert W % F == 0, f"W ({W}) not divisible by factor ({F})"

        pattern = "B C (H fH) (W fW) -> B (C fH fW) H W"
        x = einops.rearrange(z, pattern, fH=F, fW=F)
        logabsdet = z.new_zeros(B)

        return x, logabsdet


def partition(
    z: torch.Tensor,
    u_dims: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u = z[:, :u_dims]
    v = z[:, u_dims:]
    return u, v


def join(
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return torch.concat([u, v], dim=1)
