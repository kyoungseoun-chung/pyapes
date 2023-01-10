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

    mesh = Mesh(domain, None, spacing, "cpu", "double")  # type: ignore

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
