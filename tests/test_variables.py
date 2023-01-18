#!/usr/bin/env python3
"""Test mesh"""
import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import homogeneous_bcs


@pytest.mark.parametrize(
    ["domain", "spacing", "dim"],
    [
        [Box[0:1], [0.1], 1],
        [Box[0:1, 0:1], [0.1, 0.1], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_fields(domain: Box, spacing: list[float], dim: int) -> None:
    """Test field boundaries."""

    mesh = Mesh(domain, None, spacing, "cpu", "double")

    var = Field("any", 1, mesh, {"domain": None, "obstacle": None})

    test_tensor = torch.rand(
        *var.size, dtype=mesh.dtype.float, device=mesh.device
    )

    var += test_tensor
    assert_close(var(), test_tensor)

    var /= var
    assert_close(var(), torch.ones_like(test_tensor))

    var *= 10
    assert_close(var(), torch.ones_like(test_tensor) * 10)

    var -= var
    assert_close(var(), test_tensor * 0)

    # Test Field copy
    var += 2.5
    copied_var = var.copy()
    assert_close(copied_var(), test_tensor * 0 + 2.5)

    zeroed_copied_var = var.zeros_like()
    assert_close(zeroed_copied_var(), test_tensor * 0)

    copied_var_name = var.copy(name="test_copy")
    assert copied_var_name.name == "test_copy"

    zeroed_copied_var_name = var.zeros_like(name="test_zeros_like")
    assert zeroed_copied_var_name.name == "test_zeros_like"


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [0.1]],
        [Box[0:1, 0:1], [0.1, 0.1]],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1]],
    ],
)
def test_field_bcs_pad(domain: Box, spacing: list[float]) -> None:

    from pyapes.core.solver.tools import fill_pad_bc, inner_slicer, create_pad

    mesh = Mesh(domain, None, spacing, "cpu", "double")

    f_bc_d = homogeneous_bcs(mesh.dim, 0.44, "dirichlet")
    var = Field(
        "d", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    slicer = inner_slicer(mesh.dim)
    pad = create_pad(mesh.dim, 1)

    padded = fill_pad_bc(
        pad(var()[0]), 1, slicer, [var.get_bc("d-xl"), var.get_bc("d-xr")], 0
    )
    if mesh.dim == 1:
        assert_close(padded[0], padded[1])
        assert_close(padded[-1], padded[-2])
    elif mesh.dim == 2:
        assert_close(padded[0, :], padded[1, :])
        assert_close(padded[-1, :], padded[-2, :])
    else:
        assert_close(padded[0, :, :], padded[1, :, :])
        assert_close(padded[-1, :, :], padded[-2, :, :])

    f_bc_d = homogeneous_bcs(mesh.dim, 1.0, "neumann")
    var = Field(
        "d", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    padded = fill_pad_bc(
        pad(var()[0]), 1, slicer, [var.get_bc("d-xl"), var.get_bc("d-xr")], 0
    )
    if mesh.dim == 1:
        assert_close(padded[1] - padded[0], padded[2] - padded[1])
        assert_close(padded[-1] - padded[-2], padded[-2] - padded[-3])
    elif mesh.dim == 2:
        assert_close(padded[1, :] - padded[0, :], padded[2, :] - padded[1, :])
        assert_close(
            padded[-1, :] - padded[-2, :], padded[-2, :] - padded[-3, :]
        )
    else:
        assert_close(
            padded[1, :, :] - padded[0, :, :],
            padded[2, :, :] - padded[1, :, :],
        )
        assert_close(
            padded[-1, :, :] - padded[-2, :, :],
            padded[-2, :, :] - padded[-3, :, :],
        )

    f_bc_d = homogeneous_bcs(mesh.dim, None, "symmetry")
    var = Field(
        "d", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )
    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    padded = fill_pad_bc(
        pad(var()[0]), 1, slicer, [var.get_bc("d-xl"), var.get_bc("d-xr")], 0
    )

    if mesh.dim == 1:
        assert_close(padded[0], padded[2])
        assert_close(padded[-1], padded[-3])
    elif mesh.dim == 2:
        assert_close(padded[0, :], padded[2, :])
        assert_close(padded[-1, :], padded[-3, :])
    else:
        assert_close(padded[0, :, :], padded[2, :, :])
        assert_close(padded[-1, :, :], padded[-3, :, :])

    f_bc_d = homogeneous_bcs(mesh.dim, None, "periodic")
    var = Field(
        "d", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )
    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    padded = fill_pad_bc(
        pad(var()[0]), 1, slicer, [var.get_bc("d-xl"), var.get_bc("d-xr")], 0
    )
    if mesh.dim == 1:
        assert_close(padded[0], padded[-1])
        assert_close(padded[-1], padded[0])
    elif mesh.dim == 2:
        assert_close(padded[0, :], padded[-1, :])
        assert_close(padded[-1, :], padded[0, :])
    else:
        assert_close(padded[0, :, :], padded[-1, :, :])
        assert_close(padded[-1, :, :], padded[0, :, :])


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [0.1]],
        [Box[0:1, 0:1], [0.1, 0.1]],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1]],
    ],
)
def test_field_bcs(domain: Box, spacing: list[float]) -> None:
    """Test field boundaries."""

    mesh = Mesh(domain, None, spacing, "cpu", "double")

    f_bc_d = homogeneous_bcs(mesh.dim, 0.44, "dirichlet")
    var = Field(
        "d", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    if mesh.dim == 1:
        assert_close(var()[0][0].item(), 0.44)
        assert_close(var()[0][-1].item(), 0.44)
    elif mesh.dim == 2:
        assert_close(var()[0][:, 0].mean().item(), 0.44)
        assert_close(var()[0][:, -1].mean().item(), 0.44)
    else:
        assert_close(var()[0][:, :, 0].mean().item(), 0.44)
        assert_close(var()[0][:, :, -1].mean().item(), 0.44)

    f_bc_d = homogeneous_bcs(mesh.dim, 1.0, "neumann")
    var = Field(
        "n", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    if mesh.dim == 1:
        assert_close(var()[0][0], -1.0 * 0.1 + var()[0][1])
        assert_close(var()[0][-1], 1.0 * 0.1 + var()[0][-2])
    elif mesh.dim == 2:
        assert_close(var()[0][0, :], -1.0 * 0.1 + var()[0][1, :])
        assert_close(var()[0][-1, :], 1.0 * 0.1 + var()[0][-2, :])
    else:
        assert_close(var()[0][0, :, :], -1.0 * 0.1 + var()[0][1, :, :])
        assert_close(var()[0][-1, :, :], 1.0 * 0.1 + var()[0][-2, :, :])

    f_bc_d = homogeneous_bcs(mesh.dim, None, "periodic")
    var = Field(
        "p", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    if mesh.dim == 1:
        assert_close(var()[0][0], var()[0][-1])
        assert_close(var()[0][-1], var()[0][0])
    elif mesh.dim == 2:
        assert_close(var()[0][0, :], var()[0][-1, :])
        assert_close(var()[0][-1, :], var()[0][0, :])
    else:
        assert_close(var()[0][0, :, :], var()[0][-1, :, :])
        assert_close(var()[0][-1, :, :], var()[0][0, :, :])

    f_bc_d = homogeneous_bcs(mesh.dim, None, "symmetry")
    var = Field(
        "s", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random"
    )

    bc_xl = var.get_bc("d-xl")

    if bc_xl is not None:
        assert bc_xl.type == "symmetry"
        assert bc_xl.bc_id == "d-xl"

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    if mesh.dim == 1:
        assert_close(var()[0][0], var()[0][1])
        assert_close(var()[0][-1], var()[0][-2])
    elif mesh.dim == 2:
        assert_close(var()[0][0, :], var()[0][1, :])
        assert_close(var()[0][-1, :], var()[0][-2, :])
    else:
        assert_close(var()[0][0, :, :], var()[0][1, :, :])
        assert_close(var()[0][-1, :, :], var()[0][-2, :, :])
