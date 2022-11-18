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
        [Box[0:1], [0.2], 1],
        [Box[0:1, 0:1], [0.2, 0.2], 2],
        [Box[0:1, 0:1, 0:1], [0.2, 0.2, 0.2], 3],
    ],
)
def test_fvc_ops(domain: Box, spacing: list[float], dim: int) -> None:

    from pyapes.core.variables.bcs import BC_HD
    from pyapes.core.solver.fvc import FVC

    mesh = Mesh(domain, None, spacing)

    bc_val = 1.0
    dx = mesh.dx[0]

    # Field boundaries are all set to zero
    var = Field(
        "test", 1, mesh, {"domain": BC_HD(dim, bc_val), "obstacle": None}
    )

    var.set_var_tensor(mesh.X**2)

    grad = FVC.grad(var)

    # Check interior
    target = (2 * mesh.X)[~mesh.t_mask]
    assert_close(grad[0, 0, ~mesh.t_mask], target)

    if dim == 1:
        # 2 Corners
        xr_0 = (var()[0, 1] + var()[0, 0]) / 2.0 / dx
        xl_0 = bc_val / dx

        xr_n = bc_val / dx
        xl_n = (var()[0, -1] + var()[0, -2]) / 2.0 / dx

        assert_close(grad[0, 0, 0], xr_0 - xl_0)
        assert_close(grad[0, 0, -1], xr_n - xl_n)
    elif dim == 2:

        # 2 Corners
        xr_0 = (var()[0, 1, 0] + var()[0, 0, 0]) / 2.0 / dx
        xl_0 = bc_val / dx

        xr_n = bc_val / dx
        xl_n = (var()[0, -1, -1] + var()[0, -2, -1]) / 2.0 / dx

        yr_0 = (var()[0, 0, 1] + var()[0, 0, 0]) / 2.0 / dx
        yl_0 = bc_val / dx

        yr_n = bc_val / dx
        yl_n = (var()[0, -1, -1] + var()[0, -1, -2]) / 2.0 / dx

        assert_close(grad[0, 0, 0, 0], xr_0 - xl_0)
        assert_close(grad[0, 0, -1, -1], xr_n - xl_n)

        assert_close(grad[0, 1, 0, 0], yr_0 - yl_0)
        assert_close(grad[0, 1, -1, -1], yr_n - yl_n)

        # 2 Edges
        xr_0 = (var()[0, 1, 1] + var()[0, 0, 1]) / 2.0 / dx
        xl_0 = bc_val / dx

        xr_n = bc_val / dx
        xl_n = (var()[0, -1, 1] + var()[0, -2, 1]) / 2.0 / dx

        yr_0 = (var()[0, 1, 1] + var()[0, 1, 0]) / 2.0 / dx
        yl_0 = bc_val / dx

        yr_n = bc_val / dx
        yl_n = (var()[0, 1, -1] + var()[0, 1, -2]) / 2.0 / dx

        assert_close(grad[0, 0, 0, 1], xr_0 - xl_0)
        assert_close(grad[0, 0, -1, 1], xr_n - xl_n)

        assert_close(grad[0, 1, 1, 0], yr_0 - yl_0)
        assert_close(grad[0, 1, 1, -1], yr_n - yl_n)

    else:
        # 2 Corners
        xr_0 = (var()[0, 1, 0, 0] + var()[0, 0, 0, 0]) / 2.0 / dx
        xl_0 = bc_val / dx

        xr_n = bc_val / dx
        xl_n = (var()[0, -1, -1, -1] + var()[0, -2, -1, -1]) / 2.0 / dx

        yr_0 = (var()[0, 0, 1, 0] + var()[0, 0, 0, 0]) / 2.0 / dx
        yl_0 = bc_val / dx

        yr_n = bc_val / dx
        yl_n = (var()[0, -1, -1, -1] + var()[0, -1, -2, -1]) / 2.0 / dx

        zr_0 = (var()[0, 0, 0, 1] + var()[0, 0, 0, 0]) / 2.0 / dx
        zl_0 = bc_val / dx

        zr_n = bc_val / dx
        zl_n = (var()[0, -1, -1, -1] + var()[0, -1, -1, -2]) / 2.0 / dx

        assert_close(grad[0, 0, 0, 0, 0], xr_0 - xl_0)
        assert_close(grad[0, 0, -1, -1, -1], xr_n - xl_n)

        assert_close(grad[0, 1, 0, 0, 0], yr_0 - yl_0)
        assert_close(grad[0, 1, -1, -1, -1], yr_n - yl_n)

        assert_close(grad[0, 2, 0, 0, 0], zr_0 - zl_0)
        assert_close(grad[0, 2, -1, -1, -1], zr_n - zl_n)

        # 2 Edges
        xr_0 = (var()[0, 1, 1, 1] + var()[0, 0, 1, 1]) / 2.0 / dx
        xl_0 = bc_val / dx

        xr_n = bc_val / dx
        xl_n = (var()[0, -1, 1, 1] + var()[0, -2, 1, 1]) / 2.0 / dx

        yr_0 = (var()[0, 1, 1, 1] + var()[0, 1, 0, 1]) / 2.0 / dx
        yl_0 = bc_val / dx

        yr_n = bc_val / dx
        yl_n = (var()[0, 1, -1, 1] + var()[0, 1, -2, 1]) / 2.0 / dx

        zr_0 = (var()[0, 1, 1, 1] + var()[0, 1, 1, 0]) / 2.0 / dx
        zl_0 = bc_val / dx

        zr_n = bc_val / dx
        zl_n = (var()[0, 1, 1, -1] + var()[0, 1, 1, -2]) / 2.0 / dx

        assert_close(grad[0, 0, 0, 1, 1], xr_0 - xl_0)
        assert_close(grad[0, 0, -1, 1, 1], xr_n - xl_n)

        assert_close(grad[0, 1, 1, 0, 1], yr_0 - yl_0)
        assert_close(grad[0, 1, 1, -1, 1], yr_n - yl_n)

        assert_close(grad[0, 2, 1, 1, 0], zr_0 - zl_0)
        assert_close(grad[0, 2, 1, 1, -1], zr_n - zl_n)

    laplacian = FVC.laplacian(1.0, var)

    # Check interior
    target = (2 + (mesh.X) * 0.0)[~mesh.t_mask]
    assert_close(laplacian[0, ~mesh.t_mask], target)

    # Check edge values
    if dim == 1:
        # 2 Corners
        xr_0 = (var()[0, 1] - var()[0, 0]) / dx / dx
        xl_0 = (var()[0, 0] - bc_val) / (0.5 * dx) / dx

        xr_n = (bc_val - var()[0, -1]) / (0.5 * dx) / dx
        xl_n = (var()[0, -1] - var()[0, -2]) / dx / dx

        assert_close(laplacian[0, 0], xr_0 - xl_0)
        assert_close(laplacian[0, -1], xr_n - xl_n)
    elif dim == 2:
        # 2 Corners
        xr_0 = (var()[0, 1, 0] - var()[0, 0, 0]) / dx / dx
        xl_0 = (var()[0, 0, 0] - bc_val) / (0.5 * dx) / dx
        yr_0 = (var()[0, 0, 1] - var()[0, 0, 0]) / dx / dx
        yl_0 = (var()[0, 0, 0] - bc_val) / (0.5 * dx) / dx

        assert_close(laplacian[0, 0, 0], xr_0 - xl_0 + yr_0 - yl_0)

        xr_n = (bc_val - var()[0, -1, -1]) / (0.5 * dx) / dx
        xl_n = (var()[0, -1, -1] - var()[0, -2, -1]) / dx / dx
        yr_n = (bc_val - var()[0, -1, -1]) / (0.5 * dx) / dx
        yl_n = (var()[0, -1, -1] - var()[0, -1, -2]) / dx / dx

        assert_close(laplacian[0, -1, -1], xr_n - xl_n + yr_n - yl_n)

        # 2 Edges
        xr_0 = (var()[0, 2, 0] - var()[0, 1, 0]) / dx / dx
        xl_0 = (var()[0, 1, 0] - var()[0, 0, 0]) / dx / dx
        yr_0 = (var()[0, 1, 1] - var()[0, 1, 0]) / dx / dx
        yl_0 = (var()[0, 1, 0] - bc_val) / (0.5 * dx) / dx

        assert_close(laplacian[0, 1, 0], xr_0 - xl_0 + yr_0 - yl_0)

        xr_n = (bc_val - var()[0, -1, 1]) / (0.5 * dx) / dx
        xl_n = (var()[0, -1, 1] - var()[0, -2, 1]) / dx / dx
        yr_n = (var()[0, -1, 2] - var()[0, -1, 1]) / (0.5 * dx) / dx
        yl_n = (var()[0, -1, 1] - var()[0, -1, 0]) / dx / dx

        assert_close(laplacian[0, -1, 1], xr_n - xl_n + yr_n - yl_n)
    else:
        # 2 Corners
        xr_0 = (var()[0, 1, 0, 0] - var()[0, 0, 0, 0]) / dx / dx
        xl_0 = (var()[0, 0, 0, 0] - bc_val) / (0.5 * dx) / dx
        yr_0 = (var()[0, 0, 1, 0] - var()[0, 0, 0, 0]) / dx / dx
        yl_0 = (var()[0, 0, 0, 0] - bc_val) / (0.5 * dx) / dx
        zr_0 = (var()[0, 0, 0, 1] - var()[0, 0, 0, 0]) / dx / dx
        zl_0 = (var()[0, 0, 0, 0] - bc_val) / (0.5 * dx) / dx

        assert_close(
            laplacian[0, 0, 0, 0], xr_0 - xl_0 + yr_0 - yl_0 + zr_0 - zl_0
        )

        xr_n = (bc_val - var()[0, -1, -1, -1]) / (0.5 * dx) / dx
        xl_n = (var()[0, -1, -1, -1] - var()[0, -2, -1, -1]) / dx / dx
        yr_n = (bc_val - var()[0, -1, -1, -1]) / (0.5 * dx) / dx
        yl_n = (var()[0, -1, -1, -1] - var()[0, -1, -2, -1]) / dx / dx
        zr_n = (bc_val - var()[0, -1, -1, -1]) / (0.5 * dx) / dx
        zl_n = (var()[0, -1, -1, -2] - var()[0, -1, -2, -1]) / dx / dx

        assert_close(
            laplacian[0, -1, -1, -1],
            xr_n - xl_n + yr_n - yl_n + zr_n - zl_n,
        )

        # 2 Edges
        xr_0 = (var()[0, 0, 1, 2] - var()[0, 0, 1, 1]) / dx / dx
        xl_0 = (var()[0, 0, 1, 1] - var()[0, 0, 1, 0]) / dx / dx
        yr_0 = (var()[0, 0, 2, 1] - var()[0, 0, 1, 1]) / dx / dx
        yl_0 = (var()[0, 0, 1, 1] - var()[0, 0, 0, 1]) / dx / dx
        zr_0 = (var()[0, 1, 1, 1] - var()[0, 0, 1, 1]) / dx / dx
        zl_0 = (var()[0, 0, 1, 1] - bc_val) / (0.5 * dx) / dx

        assert_close(
            laplacian[0, 0, 1, 1], xr_0 - xl_0 + yr_0 - yl_0 + zr_0 - zl_0
        )

        xr_n = (bc_val - var()[0, -2, -2, -1]) / (0.5 * dx) / dx
        xl_n = (var()[0, -2, -2, -1] - var()[0, -2, -2, -2]) / dx / dx
        yr_n = (var()[0, -2, -1, -1] - var()[0, -2, -2, -1]) / dx / dx
        yl_n = (var()[0, -2, -2, -1] - var()[0, -2, -3, -1]) / dx / dx
        zr_n = (var()[0, -1, -2, -1] - var()[0, -2, -2, -1]) / dx / dx
        zl_n = (var()[0, -2, -2, -1] - var()[0, -3, -2, -1]) / dx / dx

        assert_close(
            laplacian[0, -2, -2, -1],
            xr_n - xl_n + yr_n - yl_n + zr_n - zl_n,
        )
        pass

    bc_val = 1.0

    # Test div(scalar, vector)
    # Scalar field
    var_i = Field(
        "test", 1, mesh, {"domain": BC_HD(dim, bc_val), "obstacle": None}
    )
    var_i.set_var_tensor(mesh.X)

    # Vector field
    var_j = Field(
        "test",
        3,
        mesh,
        {"domain": BC_HD(dim, 0.0), "obstacle": None},
        init_val=[1.0, 2.0, 3.0],
    )

    div = FVC.div(var_i, var_j)
    pass

    # Test div(vector, vector)

    source = FVC.source(9.81, var)
    target = torch.zeros_like(var()) + 9.81
    assert_close(source[0, 0], target)

    tensor = FVC.tensor(target)
    assert_close(tensor, target)
