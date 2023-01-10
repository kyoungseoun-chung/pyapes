#!/usr/bin/env python
"""Demo for solving the 1D Poisson equation."""
import matplotlib.pyplot as plt

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.testing.poisson import poisson_bcs
from pyapes.testing.poisson import poisson_exact_nd
from pyapes.testing.poisson import poisson_rhs_nd


def poisson():
    """poisson in N-D cases.
    Note:
        - See `pyapes.testing.poisson` for more details.
    """

    # Construct mesh
    mesh = Mesh(Box[0:1], None, [0.02])

    f_bc = poisson_bcs(1)  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None})
    rhs = poisson_rhs_nd(mesh, var)  # RHS
    sol_ex = poisson_exact_nd(mesh)  # exact solution

    solver_config = {
        "fdm": {
            "method": "cg",
            "tol": 1e-6,
            "max_it": 1000,
            "report": True,
        }
    }

    solver = Solver(solver_config)
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == fdm.rhs(rhs))
    report = solver.solve()

    _, ax = plt.subplots()

    ax.plot(
        mesh.X, sol_ex, label="Exact", marker="x", linestyle="None", color="r"
    )
    ax.plot(mesh.X, var()[0], label="Numerical", color="b", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.75, 0.9)
    ax.set_yticks([0.75, 0.8, 0.85, 0.9])
    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$p$", fontsize=16)
    ax.set_title(
        f"1D Poisson with cg solver: tol={report['tol']}, itr={report['itr']}",
        fontsize=16,
    )
    ax.legend(loc="upper left")
    plt.savefig("../assets/demo_figs/poisson_1d.png", dpi=150)


if __name__ == "__main__":
    poisson()
