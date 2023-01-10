#!/usr/bin/env python3
import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.testing.poisson import poisson_bcs
from pyapes.testing.poisson import poisson_exact_nd
from pyapes.testing.poisson import poisson_rhs_nd


@pytest.mark.parametrize(["dim"], [[1], [2], [3]])
def test_solver_tools(dim: int) -> None:
    """Testing `create_pad`, `inner_slicer` and `fill_pad` functions."""

    from pyapes.core.solver.tools import create_pad, inner_slicer, fill_pad

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

    f_bc = poisson_bcs(dim, debug=True)  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None})
    rhs = poisson_rhs_nd(mesh, var)  # RHS
    sol_ex = poisson_exact_nd(mesh)  # exact solution

    solver_config = {
        "fdm": {
            "method": "cg",
            "tol": 1e-5,
            "max_it": 1000,
            "report": True,
        }
    }

    solver = Solver(solver_config)
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == fdm.rhs(rhs))
    res, report = solver.solve()

    pass
