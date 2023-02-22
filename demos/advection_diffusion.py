#!/usr/bin/env python3
"""Demo for solving the 1D advection diffusion equation."""
from math import exp

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import homogeneous_bcs


def sol_exact(mesh, epsilon):
    return mesh.X - (torch.exp(-(1 - mesh.X) / epsilon) - exp(-1 / epsilon)) / (
        1 - exp(-1 / epsilon)
    )


def advection_diffusion(epsilon: float, mesh: Mesh) -> tuple[Tensor, Tensor, Tensor]:

    f_bc = homogeneous_bcs(1, 0.0, "dirichlet")

    # Target variable
    var = Field("U", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.5)

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-5,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.grad(var) - fdm.laplacian(epsilon, var) == 1.0)
    solver.solve()

    return var(), sol_exact(mesh, epsilon), mesh.X


if __name__ == "__main__":

    sols_numeric: list[Tensor] = []
    sols_exact: list[Tensor] = []
    epsilons: list[float] = [1, 0.5, 0.2, 0.1, 0.02]
    mesh = Mesh(Box[0:1], None, [0.02])

    for eps in epsilons:
        numeric, exact, x = advection_diffusion(eps, mesh)
        sols_numeric.append(numeric)
        sols_exact.append(exact)

    _, ax = plt.subplots()

    for idx, (n, e) in enumerate(zip(sols_numeric, sols_exact)):
        ax.plot(
            mesh.X,
            e,
            label="Exact" if idx == 0 else None,
            marker="x",
            linestyle="None",
            color="r",
        )
        ax.plot(
            mesh.X,
            n[0],
            label="Numerical" if idx == 0 else None,
            color="b",
            linestyle="--",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.text(0.5, 0.05, r"$\epsilon = 1$", fontsize=10)
    ax.text(0.55, 0.18, r"$\epsilon = 0.5$", fontsize=10)
    ax.text(0.63, 0.42, r"$\epsilon = 0.2$", fontsize=10)
    ax.text(0.72, 0.6, r"$\epsilon = 0.1$", fontsize=10)
    ax.text(0.84, 0.8, r"$\epsilon = 0.02$", fontsize=10)

    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$u$", fontsize=16)

    ax.set_title(
        "1D advection-diffusion with bicgstab solver",
        fontsize=16,
    )
    ax.legend(loc="upper left")
    plt.savefig("../assets/demo_figs/advection_diffusion_1d.png", dpi=150)
