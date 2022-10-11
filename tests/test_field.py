#!/usr/bin/env python3
"""Test mesh"""
from typing import Optional
from typing import Union

import pytest
import torch
from torch import Tensor

from pyABC.core.geometry import Box
from pyABC.core.mesh import Mesh


def create_test_field_bcs(
    bc_callable: Optional[bool], dim: int = 3
) -> Optional[list[dict[str, Union[float, str]]]]:
    """Currently we only have patches."""

    bc_config = []

    for i in range(dim * 2):

        if bc_callable is None:
            bc_val = 0
        else:
            if bc_callable:
                bc_val = bc_callable_func
            else:
                bc_val = float(i)

        bc_config.append(
            {
                "bc_obj": "patch",  # for debugging purposes
                "bc_type": "dirichlet",
                "bc_val": bc_val,
            }
        )

    return bc_config


def bc_callable_func(mesh: Mesh, mask: Tensor) -> Tensor:
    """Callable type of bc_val for test."""

    bc_val = torch.zeros_like(mesh.X[mask])

    if mesh.dim == 1:
        flag = mesh.X[mask].ge(0.5)
    elif mesh.dim == 2:
        assert mesh.Y is not None
        flag = torch.logical_and((mesh.X[mask] > 0.5), (mesh.Y[mask] > 0.5))
    elif mesh.dim == 3:
        assert mesh.Y is not None and mesh.Z is not None
        flag = torch.logical_and((mesh.X[mask] > 0.5), (mesh.Y[mask] > 0.5))
        flag = torch.logical_and(flag, (mesh.Z[mask] > 0.5))
    else:
        raise ValueError("Something wrong?")

    bc_val[flag] = 1.0

    return bc_val


@pytest.mark.parametrize(
    "domain",
    [
        (Box[0:1], None, [0.1]),
        (Box[0:1, 0:1], None, [0.1, 0.1]),
        (Box[0:1, 0:1, 0:1], None, [0.1, 0.1, 0.1]),
    ],
)
def test_field(domain: tuple) -> None:
    """Test field boundaries."""

    from pyABC.core.variables import Field

    # First, check without callable function for the bc
    f_bc_config = create_test_field_bcs(False)

    mesh = Mesh(*domain, "cpu", "double")  # type: ignore

    var = Field("any", 1, mesh, {"domain": f_bc_config, "obstacle": None})

    test_tensor = torch.rand(
        *var.size, dtype=mesh.dtype.float, device=mesh.device
    )
    var += test_tensor

    for bc, m in zip(var.bcs, var.masks):

        mask = var.masks[m]

        bc.apply(mask, var(), mesh)

    # Check bc_values
    if mesh.dim == 1:
        torch.testing.assert_close(var()[0, 0].item(), 0.0)
        torch.testing.assert_close(var()[0, -1].item(), 1.0)
    elif mesh.dim == 2:
        torch.testing.assert_close(var()[0, 1, 0].item(), 0.0)
        torch.testing.assert_close(var()[0, 1, -1].item(), 1.0)
        torch.testing.assert_close(var()[0, 0, 1].item(), 2.0)
        torch.testing.assert_close(var()[0, -1, 1].item(), 3.0)
    else:
        torch.testing.assert_close(var()[0, 0, 1, 1].item(), 0.0)
        torch.testing.assert_close(var()[0, -1, 1, 1].item(), 1.0)
        torch.testing.assert_close(var()[0, 1, 0, 1].item(), 2.0)
        torch.testing.assert_close(var()[0, 1, -1, 1].item(), 3.0)
        torch.testing.assert_close(var()[0, 1, 1, 0].item(), 4.0)
        torch.testing.assert_close(var()[0, 1, 1, -1].item(), 5.0)

    # Test for the callable BCs
    f_bc_config = create_test_field_bcs(True, dim=mesh.dim)
    # Include obstacle warning test

    mesh = Mesh(*domain, "cpu", "double")  # type: ignore

    var = Field("any", 1, mesh, {"domain": f_bc_config, "obstacle": None})

    for bc, m in zip(var.bcs, var.masks):

        mask = var.masks[m]

        bc.apply(mask, var(), mesh)

    if mesh.dim == 1:
        torch.testing.assert_close(var()[0, 0].item(), 0.0)
        torch.testing.assert_close(var()[0, -1].item(), 1.0)
    elif mesh.dim == 2:
        torch.testing.assert_close(var()[0, 0, 0].item(), 0.0)
        torch.testing.assert_close(var()[0, -1, -1].item(), 1.0)
    else:
        torch.testing.assert_close(var()[0, 0, 0, 0].item(), 0.0)
        torch.testing.assert_close(var()[0, -1, -1, -1].item(), 1.0)
