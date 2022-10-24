#!/usr/bin/env python3
"""Test fvc (finite difference - volume centered) solver module."""
from math import pi

import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyABC.core.geometry import Box
from pyABC.core.mesh import Mesh
from pyABC.core.solver.ops import Solver
from pyABC.core.variables import Field
from pyABC.testing.poisson import poisson_bcs
from pyABC.testing.poisson import poisson_exact_nd
from pyABC.testing.poisson import poisson_rhs_nd


@pytest.mark.parametrize(
    "domain",
    [
        (Box[0:1], None, [0.01]),
        (Box[0:1, 0:1], None, [64, 64]),
        (Box[0:1, 0:1, 0:1], None, [0.1, 0.1, 0.1]),
    ],
)
def test_poisson_nd(domain: tuple) -> None:
    """Test poisson in N-D cases.

    Reference:
        - 1D: https://farside.ph.utexas.edu/teaching/329/lectures/node66.html
        - 2D: https://farside.ph.utexas.edu/teaching/329/lectures/node71.html
        - 3D: Zhi Shi et al (2012) (https://doi.org/10.1016/j.apm.2011.11.078)
    """

    mesh = Mesh(*domain, "cpu", "double")  # type: ignore
    dim = mesh.dim
    f_bc_config = poisson_bcs(dim)
    var = Field("any", 1, mesh, {"domain": f_bc_config, "obstacle": None})

    solver_config = {
        "fvc": {
            "method": "cg",
            "tol": 1e-5,
            "max_it": 1000,
            "report": True,
        }
    }

    fvc = Solver(solver_config).fvc

    rhs = poisson_rhs_nd(mesh)
    sol_ex = poisson_exact_nd(mesh)

    fvc.set_eq(fvc.laplacian(var) == rhs)
    cg_sol = fvc.solve()

    assert_close(cg_sol()[0], sol_ex, rtol=0.1, atol=0.01)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_poisson_jacobi_gs_and_etc(device: str) -> None:
    """Test the Poisson Jacobi and Gauss-Seidel solver."""

    from pyABC.core.solver.fdm import fdm_op

    f_bc_config = poisson_bcs(dim=3)

    if device == "cuda":
        if not torch.cuda.is_available():
            return
    elif device == "mps":
        # MPS does not support double precision. Skip this...
        return

    mesh = Mesh(Box[0:1, 0:1, 0:1], None, [0.1, 0.1, 0.1], device, "single")

    assert mesh.dx[0] == 0.1
    assert mesh.nx[0] == 11

    var = Field("any", 1, mesh, {"domain": f_bc_config, "obstacle": None})

    assert var.size == (1, 11, 11, 11)
    assert_close(var.sum(), torch.sum(var(), dim=0))

    # RHS of the Poisson equation
    zero_tensor = torch.zeros_like(var())

    rhs = poisson_rhs_nd(mesh)
    sol_ex = poisson_exact_nd(mesh)

    # Test Jacobi method
    solver_config = {
        "method": "jacobi",
        "omega": 2 / 3,
        "tol": 1e-5,
        "max_it": 1000,
        "report": True,
    }

    # Test with copy function
    jacobi_sol, _ = fdm_op(var.copy(), rhs, solver_config, mesh)

    assert_close(jacobi_sol()[0], sol_ex, rtol=0.1, atol=0.01)

    # Test Gauss-Seidel method
    solver_config = {
        "method": "gs",
        "tol": 1e-5,
        "omega": 2 / 3,
        "max_it": 1000,
        "report": True,
    }

    solver_config = {
        "fvc": {
            "method": "cg",
            "tol": 1e-5,
            "max_it": 100,
            "report": True,
        }
    }
    # Test as a whole
    fvc = Solver(solver_config).fvc

    # Assimilate the iteration
    for _ in range(3):

        # Add noise
        rand_tensor = torch.rand(*var.size)
        var.set_var_tensor(zero_tensor + 0.0001 * rand_tensor)
        fvc.set_eq(fvc.laplacian(var) == rhs)
        sol = fvc.solve()

        assert_close(sol()[0], sol_ex, rtol=0.1, atol=0.01)
