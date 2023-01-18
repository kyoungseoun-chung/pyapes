#!/usr/bin/env python3
from math import exp
from math import pi

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import homogeneous_bcs
from pyapes.testing.burgers import burger_exact_nd
from pyapes.testing.poisson import poisson_bcs
from pyapes.testing.poisson import poisson_exact_nd
from pyapes.testing.poisson import poisson_rhs_nd


@pytest.mark.parametrize(["dim"], [[1], [2], [3]])
def test_solver_tools(dim: int) -> None:
    """Testing `create_pad`, `inner_slicer` and `fill_pad` functions."""

    from pyapes.core.mesh.tools import create_pad, inner_slicer
    from pyapes.core.solver.tools import fill_pad

    var_entry = 3
    if dim == 1:
        var = torch.rand(var_entry)
    elif dim == 2:
        var = torch.rand(var_entry, var_entry)
    else:
        var = torch.rand(var_entry, var_entry, var_entry)

    pad_1 = create_pad(dim, 1)
    pad_2 = create_pad(dim, 2)

    slicer_1 = inner_slicer(dim, 1)
    slicer_2 = inner_slicer(dim, 2)

    var_padded_1 = fill_pad(pad_1(var), dim - 1, 1, slicer_1)
    var_padded_2 = fill_pad(pad_2(var), dim - 1, 2, slicer_2)

    if dim == 1:
        assert_close(var_padded_1[0], var_padded_1[slicer_1][0])
        assert_close(var_padded_1[-1], var_padded_1[slicer_1][-1])

        assert_close((var_padded_2[:2].sum() / 2), var_padded_2[slicer_2][0])
        assert_close((var_padded_2[-2:].sum() / 2), var_padded_2[slicer_2][-1])
    elif dim == 2:
        assert_close(var_padded_1[1:-1, 0], var_padded_1[slicer_1][:, 0])
        assert_close(var_padded_1[1:-1, -1], var_padded_1[slicer_1][:, -1])

        assert_close(
            (var_padded_2[1:-1, :2].sum(dim=1)[1:-1] / 2),
            var_padded_2[slicer_2][:, 0],
        )
        assert_close(
            (var_padded_2[1:-1, -2:].sum(dim=1)[1:-1] / 2),
            var_padded_2[slicer_2][:, -1],
        )
    else:
        assert_close(
            var_padded_1[1:-1, 1:-1, 0], var_padded_1[slicer_1][:, :, 0]
        )
        assert_close(
            var_padded_1[1:-1, 1:-1, -1], var_padded_1[slicer_1][:, :, -1]
        )

        assert_close(
            (var_padded_2[1:-1, 1:-1, :2].sum(dim=2)[1:-1, 1:-1] / 2),
            var_padded_2[slicer_2][:, :, 0],
        )
        assert_close(
            (var_padded_2[1:-1, 1:-1, -2:].sum(dim=2)[1:-1, 1:-1] / 2),
            var_padded_2[slicer_2][:, :, -1],
        )


@pytest.mark.parametrize(
    ["domain", "spacing", "dim"],
    [
        [Box[0:1], [0.01], 1],
        [Box[0:1, 0:1], [0.01, 0.01], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_poisson_nd(domain: Box, spacing: list[float], dim: int) -> None:
    """Test poisson in N-D cases.
    Note:
        - See `pyapes.testing.poisson` for more details.
    """

    # Construct mesh
    mesh = Mesh(domain, None, spacing)

    f_bc = poisson_bcs(dim)  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None})
    rhs = poisson_rhs_nd(mesh, var)  # RHS
    sol_ex = poisson_exact_nd(mesh)  # exact solution

    solver = Solver(
        {"fdm": {"method": "cg", "tol": 1e-6, "max_it": 1000, "report": True}}
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == fdm.rhs(rhs))
    solver.solve()

    assert_close(var()[0], sol_ex, rtol=0.1, atol=0.01)


def test_advection_diffussion_1d() -> None:
    # Construct mesh
    mesh = Mesh(Box[0:1], None, [0.05])

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

    epsilon = 0.5

    sol_ex = mesh.X - (
        torch.exp(-(1 - mesh.X) / epsilon) - exp(-1 / epsilon)
    ) / (1 - exp(-1 / epsilon))
    solver.set_eq(fdm.grad(var) - fdm.laplacian(epsilon, var) == 1.0)
    solver.solve()

    assert_close(var()[0], sol_ex, rtol=0.1, atol=0.01)


def test_burger_1d() -> None:
    # Construct mesh
    mesh = Mesh(Box[0 : 2 * pi], None, [101])

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
    fdm = FDM({"div": {"limiter": "none"}})

    # Viscosity
    nu = 0.1
    sim_end = 0.1
    n_itr = 10
    dt = sim_end / n_itr

    res: list[Tensor] = []

    # Set dt to variable
    f_bc = homogeneous_bcs(1, None, "periodic")

    # Target variable
    init_val = burger_exact_nd(mesh, nu, 0.0)
    var = Field(
        "U", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=[init_val]
    )

    var.set_time(dt, 0.0)
    var.save_old()

    for _ in range(n_itr):

        res.append(var()[0].clone())

        solver.set_eq(
            fdm.ddt(var) + fdm.div(var, var) - fdm.laplacian(nu, var) == 0.0
        )
        solver.solve()
        var.update_time()

        sol_ex = burger_exact_nd(mesh, nu, var.t)

        assert_close(var()[0], sol_ex, rtol=0.01, atol=0.001)
