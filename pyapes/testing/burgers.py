#!/usr/bin/env python3
"""Testing module to solve the Burgers' equation.

Reference:
    - 1d: http://www.thevisualroom.com/burgers_equation.html
    - 2d and 3d: WIP
"""
from math import pi

import torch
from torch import Tensor

from pyapes.mesh import Mesh


def burger_exact_nd(mesh: Mesh, nu: float, t: float) -> Tensor:
    """Exact solution of the burgers equation from the given references."""

    if mesh.dim == 1:
        phi = torch.exp(-((mesh.X - 4 * t) ** 2) / (4 * nu * (t + 1))) + torch.exp(
            -((mesh.X - 4 * t - 2 * pi) ** 2) / (4 * nu * (t + 1))
        )

        dphi_dx = -(
            0.5
            * (mesh.X - 4 * t)
            / (nu * (t + 1))
            * torch.exp(-((mesh.X - 4 * t) ** 2) / (4 * nu * (t + 1)))
        ) - (
            0.5
            * (mesh.X - 4 * t - 2 * pi)
            / (nu * (t + 1))
            * torch.exp(-((mesh.X - 4 * t - 2 * pi) ** 2) / (4 * nu * (t + 1)))
        )

        return -2 * nu * dphi_dx / phi + 4

    elif mesh.dim == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError
