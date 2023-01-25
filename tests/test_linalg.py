#!/usr/bin/env python3
"""Test linear system solvers. Including `CG` and `BICGSTAB`.

Tests are carried out by solving the Poisson equation with different boundary conditions.
"""
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import mixed_bcs


def func_n1(grid: tuple[Tensor, ...], mask: Tensor) -> Tensor:
    """Return the value of the Neumann boundary condition (sin(5x))."""

    return -torch.sin(5.0 * grid[0][mask])


def test_2d_poisson_pure_neumann() -> None:
    """Test the Poisson equation with dirichlet and neumann boundary conditions.

    Reference:
        - https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/auto-adaptive-poisson/python/documentation.html
    """

    # Construct mesh
    mesh = Mesh(Box[0:1, 0:1], None, [21, 21])

    # xl - xr - yl - yr
    f_bc = mixed_bcs(
        [func_n1, func_n1, func_n1, func_n1],
        ["neumann", "neumann", "neumann", "neumann"],
    )  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0)
    rhs = torch.zeros_like(var())
    rhs[0] = -10 * torch.exp(
        -((mesh.X - 0.5) ** 2 + (mesh.Y - 0.5) ** 2) / 0.02
    )

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-8,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == fdm.rhs(rhs))
    solver.solve()

    # Test Neumann

    # grad_yl = (var()[0][:, 1] - var()[0][:, 0]) / mesh.dx[0]
    # grad_yr = (var()[0][:, -1] - var()[0][:, -2]) / mesh.dx[0]

    # grad_ex_yl = torch.sin(5.0 * mesh.X[:, 0])
    # grad_ex_yr = torch.sin(5.0 * mesh.X[:, -1])

    # assert_close(var()[0][0, :], torch.zeros_like(var()[0][0, :]))
    # assert_close(var()[0][-1, 1:-1], torch.zeros_like(var()[0][-1, 1:-1]))

    # assert_close(grad_yl[1:-1], grad_ex_yl[1:-1])
    # assert_close(grad_yr[1:-1], grad_ex_yr[1:-1])

    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(mesh.X, mesh.Y, var()[0], cmap=cm.coolwarm)
    plt.show()
    pass
