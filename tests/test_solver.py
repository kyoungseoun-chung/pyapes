#!/usr/bin/env python3
import pytest
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
        [Box[0:1], [0.01], 1],
        [Box[0:1, 0:1], [0.01, 0.01], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_poisson_nd(domain: Box, spacing: list[float], dim: int) -> None:
    """Test poisson in N-D cases.

    Reference:
        - 1D: https://farside.ph.utexas.edu/teaching/329/lectures/node66.html
        - 2D: https://farside.ph.utexas.edu/teaching/329/lectures/node71.html
        - 3D: Zhi Shi et al (2012) (https://doi.org/10.1016/j.apm.2011.11.078)
    """

    mesh = Mesh(domain, None, spacing)
    f_bc = poisson_bcs(dim)
    var = Field("test", 1, mesh, {"domain": f_bc, "obstacle": None})

    solver_config = {
        "fvc": {
            "method": "cg",
            "tol": 1e-5,
            "max_it": 1000,
            "report": True,
        }
    }

    solver = Solver(solver_config)
    fvc = solver.fvc
    fvm = solver.fvm

    rhs = poisson_rhs_nd(mesh)
    sol_ex = poisson_exact_nd(mesh)

    solver.set_eq(fvm.laplacian(1.0, var) == fvc.tensor(rhs))
    cg_sol = solver()

    assert_close(cg_sol()[0], sol_ex, rtol=0.1, atol=0.01)
