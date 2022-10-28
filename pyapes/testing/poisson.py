#!/usr/bin/env python3
"""Collection of boundary conditions and solution of the Poisson equation.

Reference:
        - 1D: https://farside.ph.utexas.edu/teaching/329/lectures/node66.html
        - 2D: https://farside.ph.utexas.edu/teaching/329/lectures/node71.html
        - 3D: Zhi Shi et al (2012) (https://doi.org/10.1016/j.apm.2011.11.078)
"""
from math import pi
from typing import Union

import torch
from torch import Tensor

from pyapes.core.mesh import Mesh


def poisson_rhs_nd(mesh: Mesh) -> Tensor:
    """RHS of the poisson equation given by the references."""

    if mesh.dim == 1:
        return 1.0 - 2.0 * mesh.X**2
    elif mesh.dim == 2:
        return 6.0 * mesh.X * mesh.Y * (1.0 - mesh.Y) - 2.0 * (mesh.X**3)
    else:
        return (
            torch.sin(pi * mesh.X)
            * torch.sin(pi * mesh.Y)
            * torch.sin(pi * mesh.Z)
        )


def poisson_exact_nd(mesh: Mesh) -> Tensor:
    """Exact solution of the poisson equation from the given references."""

    if mesh.dim == 1:
        return (
            7.0 / 9.0
            - 2.0 / 9.0 * mesh.X
            + mesh.X**2 / 2.0
            - mesh.X**4 / 6.0
        )
    elif mesh.dim == 2:
        return mesh.Y * (1.0 - mesh.Y) * (mesh.X**3)
    else:
        return (
            -1.0
            / (3 * pi**2)
            * torch.sin(pi * mesh.X)
            * torch.sin(pi * mesh.Y)
            * torch.sin(pi * mesh.Z)
        )


def poisson_bcs(dim: int = 3) -> list[dict[str, Union[float, str]]]:
    """Currently we only have patches."""

    bc_config = []

    for _ in range(dim * 2):
        if dim == 1:
            bc_val = poisson_1d_bc
        elif dim == 2:
            bc_val = poisson_2d_bc
        else:
            bc_val = 0.0

        bc_config.append(
            {
                "bc_obj": "patch",  # for debugging purposes
                "bc_type": "dirichlet",
                "bc_val": bc_val,
            }
        )

    return bc_config


def poisson_1d_bc(mesh: Mesh, mask: Tensor) -> Tensor:

    return (
        7.0 / 9.0
        - 2.0 / 9.0 * mesh.X[mask]
        + mesh.X[mask] ** 2 / 2.0
        - mesh.X[mask] ** 4 / 6.0
    )


def poisson_2d_bc(mesh: Mesh, mask: Tensor) -> Tensor:

    return mesh.Y[mask] * (1.0 - mesh.Y[mask]) * (mesh.X[mask] ** 3)