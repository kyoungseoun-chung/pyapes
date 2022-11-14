#!/usr/bin/env python3
"""Test fvc (finite difference - volume centered) solver module.

Need to be revised. I comfused concept of fvc and fvm.
"""
import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.variables import Field


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
    from pyapes.core.solver.fvc import FVC

    mesh = Mesh(domain, None, spacing)

    # Field boundaries are all set to zero
    var = Field("test", 1, mesh, {"domain": BC_HD(dim, 0.0), "obstacle": None})

    var.set_var_tensor(mesh.X**2)

    grad = FVC.grad(var)

    target = (2 * mesh.X)[~mesh.t_mask]
    assert_close(grad[0, 0, ~mesh.t_mask], target)

    laplacian = FVC.laplacian(1.0, var)
    target = (2 + (mesh.X) * 0.0)[~mesh.t_mask]
    assert_close(laplacian[0, 0, ~mesh.t_mask], target)

    source = FVC.source(9.81, var)
    target = torch.zeros_like(var()) + 9.81
    assert_close(source[0, 0], target)

    tensor = FVC.tensor(target)
    assert_close(tensor, target)
