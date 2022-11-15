#!/usr/bin/env python3
"""Test mesh"""
from typing import Optional

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.geometry.basis import FDIR
from pyapes.core.geometry.basis import NUM_TO_DIR
from pyapes.core.mesh import Mesh
from pyapes.core.variables import Field
from pyapes.core.variables import Flux
from pyapes.core.variables.bcs import BC_config_type


def create_test_field_bcs(
    bc_callable: Optional[bool] = False, dim: int = 3
) -> list[BC_config_type]:
    """Currently we only have patches."""

    bc_config = []

    for i in range(dim * 2):

        if bc_callable is None:
            bc_val = 1.0
        else:
            if bc_callable:
                bc_val = bc_callable_func
            else:
                bc_val = float(i)

        bc_config.append(
            {
                "bc_face": FDIR[i],
                "bc_type": "dirichlet",
                "bc_val": bc_val,
            }
        )

    return bc_config


def bc_callable_func(grid: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Callable type of bc_val for test."""

    bc_val = torch.zeros_like(grid[0][mask])

    if dim == 1:
        flag = grid[0][mask].ge(0.5)
    elif dim == 2:
        assert grid[1] is not None
        flag = torch.logical_and((grid[0][mask] > 0.5), (grid[1][mask] > 0.5))
    else:
        assert grid[1] is not None and grid[2] is not None
        flag = torch.logical_and((grid[0][mask] > 0.5), (grid[1][mask] > 0.5))
        flag = torch.logical_and(flag, (grid[2][mask] > 0.5))

    bc_val[flag] = True

    return bc_val


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

    # First, check without callable function for the bc
    f_bc_config = create_test_field_bcs(None, dim)

    mesh = Mesh(domain, None, spacing, "cpu", "double")  # type: ignore

    var = Field("any", 1, mesh, {"domain": f_bc_config, "obstacle": None})

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

    assert len(var.bcs) == 2 * dim
    assert var.bcs[0].__class__.__name__ == "Dirichlet"


@pytest.mark.parametrize(
    ["domain", "spacing", "dim"],
    [
        [Box[0:1], [0.1], 1],
        [Box[0:1, 0:1], [0.1, 0.1], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_fluxes(domain: Box, spacing: list[float], dim: int) -> None:

    # First, check without callable function for the bc
    f_bc_config = create_test_field_bcs(False, dim)

    mesh = Mesh(domain, None, spacing, "cpu", "double")  # type: ignore
    var = Field("any", 1, mesh, {"domain": f_bc_config, "obstacle": None})

    flux = Flux(mesh)

    zero_tensor = torch.zeros_like(mesh.X)

    # Test flux face assigning
    for i in range(dim):
        flux.to_face(0, NUM_TO_DIR[i], "l", zero_tensor + 5)
        flux.to_face(0, NUM_TO_DIR[i], "r", zero_tensor - 2)

    # Test sum
    flux.sum()

    assert_close(flux.face(0, "xl"), zero_tensor + 5)
    assert_close(flux.face(0, "xr"), zero_tensor - 2)
    assert_close(
        flux.center(0, "x"), (zero_tensor - 7) * mesh.A["xl"] / mesh.V
    )

    for i in range(dim):
        flux.to_center(0, NUM_TO_DIR[i], zero_tensor + 10)

    assert_close(flux.center(0, "x"), zero_tensor + 10)

    # Test bcs
    for bc in var.bcs:
        bc.apply(var(), flux, var.mesh.grid, 0)

    flux.sum()

    mask = mesh.d_mask["xl"]
    mask_tensor = torch.ones_like(mask)
    assert_close(
        flux.center(0, "x")[mask],
        -2 * mask_tensor[mask] * mesh.A["xl"][mask] / mesh.V[mask],
    )
    mask = mesh.d_mask["xr"]
    mask_tensor = torch.ones_like(mask)
    assert_close(
        flux.center(0, "x")[mask],
        -4 * mask_tensor[mask] * mesh.A["xl"][mask] / mesh.V[mask],
    )