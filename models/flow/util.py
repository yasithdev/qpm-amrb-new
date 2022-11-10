import torch

from . import FlowTransform

# --------------------------------------------------------------------------------------------------------------------------------------------------


def log_p_z(z: torch.Tensor, z_dist: torch.distributions.Distribution) -> torch.Tensor:
    """
    Given a base distribution and a latent-space point, return the log probability of that point
    :param z: latent-space point
    :param z_dist: base distribution
    :return: log probability of the latent-space point
    """
    return z_dist.log_prob(z)


# --------------------------------------------------------------------------------------------------------------------------------------------------


def log_p_x(
    flow: FlowTransform,
    x: torch.Tensor,
    z_dist: torch.distributions.Distribution,
) -> torch.Tensor:
    """
    Given a base distribution and a data-space point, return the log probability of that point
    :param flow: Normalizing flow from Rd -> Rd
    :param x: data-space point
    :param z_dist: base distribution
    :return: log probability of the data-space point, given the flow
    """
    z, log_det = flow.forward(x)
    return log_p_z(z, z_dist) + log_det


# --------------------------------------------------------------------------------------------------------------------------------------------------


def proj(
    z: torch.Tensor,
    manifold_dims: int,
) -> torch.Tensor:
    """
    Project Density Tensor from Ambient Space into Manifold Space

    :param z: Density in Ambient Space
    :param manifold_dims: Dimensions of Manifold Space
    :return: Projection of Density from Ambient Space into Manifold Space
    """
    zu = z[:, :manifold_dims]
    return zu


# --------------------------------------------------------------------------------------------------------------------------------------------------


def pad(
    zu: torch.Tensor,
    off_manifold_dims: int,
) -> torch.Tensor:
    """
    Project Density Tensor from Manifold Space into Ambient Space

    :param zu: Density in Manifold Space
    :param off_manifold_dims: Dimensions of Off-Manifold Space
    :param chw: Tuple of [C, H, W]
    :return: Projection of Density from Manifold Space into Ambient Space
    """
    z = torch.nn.functional.pad(zu, (0, 0, 0, 0, 0, off_manifold_dims))
    return z


# --------------------------------------------------------------------------------------------------------------------------------------------------
