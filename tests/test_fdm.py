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
    ["domain", "spacing"],
    [
        [Box[0:1], [0.2]],
        [Box[0:1, 0:1], [0.2, 0.2]],
        [Box[0:1, 0:1, 0:1], [0.2, 0.2, 0.2]],
    ],
)
def test_fdc_ops(domain: Box, spacing: list[float]) -> None:
    """Test FDC module that discretizes current field values."""

    from pyapes.core.solver.fdc import FDC

    mesh = Mesh(domain, None, spacing)

    # Field boundaries are all set to zero
    var = Field("test", 1, mesh, None)

    var.set_var_tensor(mesh.X**2)

    fdc = FDC({"div": {"limiter": "none"}})

    grad = fdc.grad(var)

    # Check interior
    target = (2 * mesh.X)[~mesh.t_mask]
    assert_close(grad[0][0][~mesh.t_mask], target)

    laplacian = fdc.laplacian(1.0, var)

    # Check interior
    target = (2 + (mesh.X) * 0.0)[~mesh.t_mask]
    assert_close(laplacian[0][~mesh.t_mask], target)

    # Test div(scalar, vector)

    # Scalar advection speed
    var_i = Field("test", 1, mesh, None)
    var_i.set_var_tensor(torch.rand(var_i().shape))
    var_j = Field("test", 1, mesh, None, init_val=2.0)

    div = fdc.div(var_i, var_j)
    dx = mesh.dx

    target = (
        (torch.roll(var_i()[0], -1, 0) - torch.roll(var_i()[0], 1, 0))
        / (2 * dx[0])
        * 2.0
    )

    assert_close(div[0][~mesh.t_mask], target[~mesh.t_mask])

    fdc.update_config("div", "limiter", "upwind")

    div = fdc.div(var_i, var_j)

    target = (var_i()[0] - torch.roll(var_i()[0], 1, 0)) / dx[0] * 2.0

    assert_close(div[0][~mesh.t_mask], target[~mesh.t_mask])


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [0.2]],
        [Box[0:1, 0:1], [0.2, 0.2]],
        [Box[0:1, 0:1, 0:1], [0.2, 0.2, 0.2]],
    ],
)
def test_solver_fdm_ops(domain: Box, spacing: list[float]) -> None:
    """Test FDM module."""

    from pyapes.core.solver.fdm import FDM

    mesh = Mesh(domain, None, spacing)

    # Field boundaries are all set to zero
    var = Field("test", 1, mesh, None)

    var.set_var_tensor(mesh.X**2)

    fdm = FDM()
