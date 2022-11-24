from typing import Optional, Tuple

import einops
import torch
import torch.nn.functional as F

from . import FlowTransform


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

    def forward(
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

    def inverse(
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


class Projection(FlowTransform):
    def __init__(
        self,
        ambient_dims: int,
        manifold_dims: int,
    ) -> None:

        super().__init__()
        self.ambient_dims = ambient_dims
        self.manifold_dims = manifold_dims

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)

        z = x.narrow(1, 0, self.manifold_dims)
        logabsdet = x.new_zeros(B)

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = z.size(0)

        padding_dims = self.ambient_dims - self.manifold_dims
        padding_shape = (0, 0) * (z.dim() - 2) + (0, padding_dims)

        x = F.pad(z, padding_shape)
        logabsdet = z.new_zeros(B)

        return x, logabsdet
