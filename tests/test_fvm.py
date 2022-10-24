#!/usr/bin/env python3
import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyABC.core.geometry import Box
from pyABC.core.mesh import Mesh
from pyABC.core.solver.ops import Solver
from pyABC.core.variables import Field


@pytest.mark.parametrize(
    ("domain", "spacing", "dim"),
    [
        (Box[0:1], [0.01], 1),
        (Box[0:1, 0:1], [64, 64], 2),
        (Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3),
    ],
)
def test_poisson(domain: Box, spacing: list[int], dim: int) -> None:

    from pyABC.testing.poisson import (
        poisson_bcs,
        poisson_rhs_nd,
        poisson_exact_nd,
    )

    mesh = Mesh(domain, None, spacing)
    var = Field("", dim, mesh, {"domain": poisson_bcs(dim), "obstacle": None})

    solver_config = {
        "fvm": {
            "method": "cg",
            "tol": 1e-5,
            "max_it": 1000,
            "report": True,
        }
    }

    fvm = Solver(solver_config).fvm
    fvm.set_eq(fvm.laplacian(1.0, var) == poisson_rhs_nd(mesh))

    assert fvm.laplacian.coeff == 1.0
    pass


@pytest.mark.parametrize(
    ("domain", "spacing", "dim"),
    [
        (Box[0:1], [5], 1),
        (Box[0:1, 0:1], [5, 5], 2),
        (Box[0:1, 0:1, 0:1], [5, 5, 5], 3),
    ],
)
def test_grad(domain: Box, spacing: list[int], dim: int) -> None:

    from pyABC.core.solver.fvm import Grad, DIR

    mesh = Mesh(domain, None, spacing)
    var = Field("", dim, mesh, None)

    if dim == 1:
        var.set_var_tensor(mesh.X**2)
    elif dim == 2:
        var.set_var_tensor(mesh.Y**2)
    else:
        var.set_var_tensor(mesh.Z**2)

    fvm_grad = Grad()
    res = fvm_grad(var)

    assert res.c_idx[0] == [*range(dim)]
    assert res.c_idx[1] == [DIR[i] for i in range(dim)]

    if dim == 1:
        assert_close(res(0, "x")[1:-1], 2 * mesh.X[1:-1])
    elif dim == 2:
        assert_close(res(0, "y")[1:-1, 1:-1], 2 * mesh.Y[1:-1, 1:-1])
    else:
        assert_close(
            res(0, "z")[1:-1, 1:-1, 1:-1], 2 * mesh.Z[1:-1, 1:-1, 1:-1]
        )


@pytest.mark.parametrize(
    ("domain", "spacing", "dim"),
    [
        (Box[0:1], [5], 1),
        (Box[0:1, 0:1], [5, 5], 2),
        (Box[0:1, 0:1, 0:1], [5, 5, 5], 3),
    ],
)
def test_fvm(domain: Box, spacing: list[int], dim: int) -> None:
    from pyABC.core.solver.fvm import Div, Laplacian

    mesh = Mesh(domain, None, spacing)
    var = Field("", dim, mesh, None)
    var.set_var_tensor(var.mesh.X)

    div = Div()
    laplacian = Laplacian()

    flux = div(var, var).flux
    sign_test = flux(0, "xl").clone()
    flux *= -1

    assert_close(flux(0, "xl"), -sign_test)

    # Also test sum of operations
    op_sum = div(var, var) - laplacian(1.0, var) + div(var, var) == 10

    assert len(op_sum.ops) == 3
    assert op_sum.ops[0]["op"] == "Div"
    assert op_sum.ops[1]["op"] == "Laplacian"

    assert_close(op_sum.rhs, torch.zeros_like(var()) + 10)
