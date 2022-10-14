#!/usr/bin/env python3
import pytest
import torch

from pyABC.core.geometry import Box
from pyABC.core.mesh import Mesh
from pyABC.core.variables import Field


@pytest.mark.parametrize(
    ("domain", "spacing", "dim"),
    [
        (Box[0:1], [5], 1),
        (Box[0:1, 0:1], [5, 5], 2),
        (Box[0:1, 0:1, 0:1], [5, 5, 5], 3),
    ],
)
def test_grad(domain: Box, spacing: list[int], dim: int) -> None:

    from pyABC.core.solver.fvm import fvm_grad, DIR

    mesh = Mesh(domain, None, spacing)
    var = Field("", dim, mesh, None)

    if dim == 1:
        var.set_var_tensor(mesh.X**2)
    elif dim == 2:
        var.set_var_tensor(mesh.Y**2)
    else:
        var.set_var_tensor(mesh.Z**2)

    res = fvm_grad(var)

    assert res.index[0] == [*range(dim)]
    assert res.index[1] == [DIR[i] for i in range(dim)]

    if dim == 1:
        torch.testing.assert_close(res(0, "x")[1:-1], 2 * mesh.X[1:-1])
    elif dim == 2:
        torch.testing.assert_close(
            res(0, "y")[1:-1, 1:-1], 2 * mesh.Y[1:-1, 1:-1]
        )
    else:
        torch.testing.assert_allclose(
            res(0, "z")[1:-1, 1:-1, 1:-1], 2 * mesh.Z[1:-1, 1:-1, 1:-1]
        )
