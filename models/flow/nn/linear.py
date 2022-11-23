from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.linalg import solve_triangular

from . import FlowTransform


class LULinear(FlowTransform):

    """
    A linear transform where weights are parameterized by LU decomposition

    Cost:
        weight: O(D^3)
        weight_inverse: O(D^3)
        forward(): [O(D^2N), O(D)]
        inverse(): [O(D^2N), O(D)]

    Where:
        D: num_features
        N: num_inputs
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        identity_init: bool = True,
    ) -> None:

        super().__init__()

        self.num_features = num_features
        self.eps = eps

        # define lower, diagonal, and upper indices
        self.tril_indices = np.tril_indices(num_features, k=-1)
        self.triu_indices = np.triu_indices(num_features, k=1)
        self.diag_indices = np.diag_indices(num_features)

        n_tri_entries = ((num_features - 1) * num_features) // 2

        # model parameters
        self.weight_l = torch.nn.Parameter(torch.zeros(n_tri_entries))
        self.weight_u = torch.nn.Parameter(torch.zeros(n_tri_entries))
        self.weight_d = torch.nn.Parameter(torch.zeros(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(1))

        if identity_init:
            torch.nn.init.zeros_(self.weight_l)
            torch.nn.init.zeros_(self.weight_u)
            torch.nn.init.constant_(self.weight_d, np.log(np.exp(1 - eps) - 1))
        else:
            stdv = 1.0 / np.sqrt(num_features)
            torch.nn.init.uniform_(self.weight_l, -stdv, stdv)
            torch.nn.init.uniform_(self.weight_u, -stdv, stdv)
            torch.nn.init.uniform_(self.weight_d, -stdv, stdv)

    @property
    def upper_diag(self) -> torch.Tensor:
        return F.softplus(self.weight_d) + self.eps

    @property
    def L(self) -> torch.Tensor:
        lower = self.weight_l.new_zeros(self.num_features, self.num_features)
        lower[self.tril_indices[0], self.tril_indices[1]] = self.weight_l
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0
        return lower

    @property
    def U(self) -> torch.Tensor:
        upper = self.weight_u.new_zeros(self.num_features, self.num_features)
        upper[self.triu_indices[0], self.triu_indices[1]] = self.weight_u
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag
        return upper

    @property
    def weight(self) -> torch.Tensor:
        return self.L @ self.U

    @property
    def weight_inverse(self) -> torch.Tensor:
        # solve LU[W_]=I by first solving L[L_]=I, and then solving U[W_]=L_
        I = torch.eye(self.num_features, self.num_features)
        L_ = solve_triangular(self.L, I, upper=False, unitriangular=True)
        W_ = solve_triangular(self.U, L_, upper=True, unitriangular=False)
        return W_

    @property
    def logabsdet(self) -> torch.Tensor:
        return torch.sum(torch.log(self.upper_diag))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)

        z = F.linear(x, self.weight)
        z = z + self.bias
        logabsdet = x.new_ones(B) * self.logabsdet

        return z, logabsdet

    def inverse(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = z.size(0)

        z = z - self.bias
        x = F.linear(z, self.weight_inverse)
        logabsdet = z.new_ones(B) * -self.logabsdet

        return x, logabsdet
