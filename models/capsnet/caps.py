"""
Implementation of 'Dynamic Routing Between Capsules'

"""

from typing import Tuple, Union

import einops
import torch


def Σ(dim, x):
    return torch.sum(x, dim=dim, keepdim=True)


def Π(dim, x):
    return torch.prod(x, dim=dim, keepdim=True)


def squash(
    s: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Activation function to squash vectors into magnitude [0,1]

    Args:
        x (torch.Tensor): input tensor (B, C, D, *)
        dim (int): axis to squash

    Returns:
        torch.Tensor: output tensor (B, C, D, *)
    """
    # compute ||s|| and clamp it
    ε = 1e-3
    s_norm = torch.norm(s, p=2, dim=dim, keepdim=True)
    s_norm = torch.clamp(s_norm, min=ε)
    # return v = s * tanh(||s||)/||s||
    return s * (torch.tanh(s_norm) / s_norm)


class ConvCaps2D(torch.nn.Module):
    """
    2D Convolution Capsule Layer.

    Op: (B, C, D, H, W) -> (B, c, d, h, w)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count
    H/W: Input Height/Width

    c: Output capsule channels
    d: Output capsule count
    h/w: Output Height/Width

    """

    def __init__(
        self,
        in_capsules: Tuple[int, int],  # (C, D)
        out_capsules: Tuple[int, int],  # (c, d)
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: str = "valid",
    ) -> None:
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        k = kernel_size
        k = (k, k) if isinstance(k, int) else k
        s = stride
        self.stride = (s, s) if isinstance(s, int) else s
        self.padding = padding
        C, D = in_capsules
        c, d = out_capsules
        self.groups = D
        self.conv_k = torch.nn.Parameter(torch.randn(d * c, C, *k))
        self.conv_b = torch.nn.Parameter(torch.randn(d * c))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, D*C, H, W)
        s = einops.rearrange(
            x,
            "B C D H W -> B (D C) H W",
            C=self.in_capsules[0],
            D=self.in_capsules[1],
        )
        # (B, D*C, H, W) -> (B, d*c, h, w)
        s = torch.nn.functional.conv2d(
            input=s,
            weight=self.conv_k,
            bias=self.conv_b,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        # (B, d*c, h, w) -> (B, c, d, h, w)
        s = einops.rearrange(
            s,
            "B (d c) h w -> B c d h w",
            c=self.out_capsules[0],
            d=self.out_capsules[1],
        )
        return s


class LinearCapsDR(torch.nn.Module):
    """
    Linear Capsule Layer with Dynamic Routing.

    Op: (B, C, i) -> (B, c, j)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    i: Input capsule count

    c: Output capsule channels
    j: Output capsule count

    """

    def __init__(
        self,
        in_capsules: Tuple[int, int],  # (C, i)
        out_capsules: Tuple[int, int],  # (c, j)
        routing_iters: int = 3,
        groups: int = 64,
    ):
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.routing_iters = routing_iters
        self.groups = groups
        C, i = in_capsules
        c, j = out_capsules
        G = groups
        assert C % G == 0
        assert c % G == 0
        C //= G
        c //= G
        # weights (j, i, c, C)
        self.weight = torch.nn.Parameter(torch.randn(G, j, i, c, C))
        self.nc = c**0.5

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        G = self.groups
        i = self.in_capsules[-1]
        j = self.out_capsules[-1]
        # input tensor (B[CG]i) -> (BGiC)
        x = einops.rearrange(x, "B (C G) i -> B G i C", G=G)
        # prediction tensor u_{j|i} (BGjic)
        u = torch.einsum("BGiC,GjicC->BGjic", x, self.weight)
        # coupling logits (BGji)
        b = x.new_zeros(B, G, j, i)
        # dynamic routing
        v = None
        for _ in range(self.routing_iters):
            c = b.softmax(dim=-2)
            # initial capsule output (BGjc)
            s = torch.einsum("BGjic,BGji->BGjc", u, c)
            v = squash(s, dim=-1)
            # add agreement to coupling logits: dot_product(v,u) (BGji)
            b = b + torch.einsum("BGjc,BGjic->BGji", v, u)
        # output tensor
        v = einops.rearrange(v, "B G j c -> B (c G) j")
        assert v is not None
        return v


class MatrixCapsEM(torch.nn.Module):
    r"""
    Matrix Capsule Layer with EM Routing.

    Op: (B, D, i) -> (B, D, j)

    Terms
    -----
    B: Batch Size
    D: Spatial Dims
    G: Number of Groups
    i: Input capsule count
    j: Output capsule count

    Operation
    ---------
    (B, D, i) -> (B, G, i) -> [** EM Routing **] -> (B, G, j) -> (B, D, j)
                                /                   \
                                

    """

    def __init__(
        self,
        matrix_size: int,
        in_capsules: int,  # i
        out_capsules: int,  # j
        routing_iters: int = 3,
    ):
        super().__init__()
        assert routing_iters > 0
        self.matrix_size = matrix_size
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.routing_iters = routing_iters
        i = in_capsules
        j = out_capsules
        M = matrix_size
        # weights (i, j, M, m)
        self.weight = torch.nn.Parameter(torch.randn(i, j, M, M))
        self.βa = torch.nn.Parameter(torch.as_tensor(1.0))
        self.βu = torch.nn.Parameter(torch.as_tensor(1.0))

    def forward(
        self,
        xi: torch.Tensor,
    ) -> torch.Tensor:
        M = self.matrix_size
        j = self.out_capsules
        xi = xi.permute(0, 2, 1)
        B, i, P = xi.shape
        assert i == self.in_capsules
        assert P == M**2 + 1
        # pose matrix, activation
        pi, ai = xi.split([M**2, 1], dim=-1)
        piMM = pi.view(B, i, M, M)
        ai11 = ai.view(B, i, 1, 1)
        # coupling logits (initially uniform)
        Rij1 = xi.new_ones(B, i, j, 1) / j
        # prediction tensor v_{j|i} = (Bijh)
        vijh = torch.einsum("BiMm,ijmN->BijMN", piMM, self.weight).reshape(B, i, j, M**2)
        # EM routing
        λ = 1.0
        π = torch.pi
        a1j1 = None
        for _ in range(self.routing_iters):
            # m step
            Rij1 = Rij1 * ai11  # Bij1
            µ1jh = Σ(1, Rij1 * vijh) / Σ(1, Rij1)  # B1jh
            σ1jh_sq = Σ(1, Rij1 * (vijh - µ1jh).square()) / Σ(1, Rij1) + 1e-4  # B1jh
            c1jh = (self.βu + σ1jh_sq.sqrt().log()) * Σ(1, Rij1)  # B1jh
            a1j1 = (self.βa - Σ(3, c1jh)).mul(λ).sigmoid()  # B1j1
            # e step
            pij1 = Σ(3, -(vijh - µ1jh).square() / (2 * σ1jh_sq)).exp() / Π(3, 2 * π * σ1jh_sq).sqrt()  # Bij1
            tmp = a1j1 * pij1  # Bij1
            tmp = tmp / Σ(2, tmp)
            Rij1 = tmp.view(B, i, j, 1)
        assert a1j1 is not None
        pj = torch.einsum("Bijh,Bijh->Bjh", vijh, Rij1).view(B, j, M * M)
        aj = a1j1.view(B, j, 1)
        xj = torch.cat([pj, aj], dim=-1)
        xj = xj.permute(0, 2, 1)
        return xj


class MaskCaps(torch.nn.Module):
    """
    Layer to obtain the capsule logits and latent.
    (No Parameters)

    The latent contain features from all capsules.
    However, only the highest-norm capsule will have non-zero features.

    Op: (B, C, D) -> [logits, latent]

    logits: (B, D)
    latent: (B, C * D)

    Terms
    -----
    B: Batch Size

    C: Input capsule channels
    D: Input capsule count

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute logits via capsule norm (B, D)
        logits = x.norm(p=2, dim=1)

        # Mask most-activated capsule output (B, C * D)
        (B, C, D) = x.size()
        I = torch.eye(D, device=x.device)
        mask = I.index_select(
            dim=0,
            index=logits.argmax(dim=1),
        )
        # (B, D)
        latent = (x * mask.reshape(B, 1, D)).reshape(B, C * D)
        # (B, C * D)

        return logits, latent
