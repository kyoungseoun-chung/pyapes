#!/usr/bin/env python3
"""Test fvc (finite difference - volume centered) solver module.

Need to be revised. I comfused concept of fvc and fvm.
"""
from math import pi

import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.testing.poisson import poisson_bcs
from pyapes.testing.poisson import poisson_exact_nd
from pyapes.testing.poisson import poisson_rhs_nd


@pytest.mark.parametrize(
    ["domain", "spacing", "dim"],
    [
        [Box[0:1], [0.1], 1],
        [Box[0:1, 0:1], [0.1, 0.1], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_fvc_ops(domain: Box, spacing: list[float], dim: int) -> None:

    from pyapes.core.variables.bcs import BC_HD
    from pyapes.core.solver.fvc import Grad

    mesh = Mesh(domain, None, spacing)

    # Field boundaries are all set to zero
    var = Field("test", 1, mesh, {"domain": BC_HD(dim, 0.0), "obstacle": None})

    var.set_var_tensor(mesh.X**2)

    grad = Grad()(var)

    target = (2 * mesh.X)[~mesh.t_mask]
    assert_close(grad(0, "x")[~mesh.t_mask], target)


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
