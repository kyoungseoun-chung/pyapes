#!/usr/bin/env python3
"""Collection of boundary conditions and solution of the Poisson equation.

Reference:
        - 1D: https://farside.ph.utexas.edu/teaching/329/lectures/node66.html
        - 2D: https://farside.ph.utexas.edu/teaching/329/lectures/node71.html
        - 3D: Zhi Shi et al (2012) (https://doi.org/10.1016/j.apm.2011.11.078)
"""
from math import pi

import torch
from torch import Tensor

from pyapes.geometry.basis import FDIR
from pyapes.mesh import Mesh
from pyapes.variables import Field
from pyapes.variables.bcs import BCConfig


def poisson_rhs_nd(mesh: Mesh, var: Field) -> Tensor:
    """RHS of the poisson equation given by the references."""

    rhs = torch.zeros_like(var())

    if mesh.dim == 1:
        rhs[0] = 1.0 - 2.0 * mesh.X**2
    elif mesh.dim == 2:
        rhs[0] = 6.0 * mesh.X * mesh.Y * (1.0 - mesh.Y) - 2.0 * (mesh.X**3)
    else:
        rhs[0] = (
            torch.sin(pi * mesh.X) * torch.sin(pi * mesh.Y) * torch.sin(pi * mesh.Z)
        )
    return rhs


def poisson_exact_nd(mesh: Mesh) -> Tensor:
    """Exact solution of the poisson equation from the given references."""

    if mesh.dim == 1:
        return 7.0 / 9.0 - 2.0 / 9.0 * mesh.X + mesh.X**2 / 2.0 - mesh.X**4 / 6.0
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


def poisson_bcs(dim: int = 3, debug: bool = False) -> list[BCConfig]:
    """Construct boundary configuration for the N-D Poisson equation."""

    bc_config = []

    for i in range(dim * 2):
        if dim == 1:
            bc_val = poisson_1d_bc
        elif dim == 2:
            bc_val = poisson_2d_bc
        else:
            bc_val = 0.0

        bc_config.append(
            {
                "bc_face": FDIR[i],  # for debugging purposes
                "bc_type": "dirichlet",
                "bc_val": 4.44 if debug else bc_val,
            }
        )

    return bc_config


def poisson_1d_bc(grid: Tensor, mask: Tensor, *_) -> Tensor:
    return (
        7.0 / 9.0
        - 2.0 / 9.0 * grid[0][mask]
        + grid[0][mask] ** 2 / 2.0
        - grid[0][mask] ** 4 / 6.0
    )


def poisson_2d_bc(grid: Tensor, mask: Tensor, *_) -> Tensor:
    return grid[1][mask] * (1.0 - grid[1][mask]) * (grid[0][mask] ** 3)
