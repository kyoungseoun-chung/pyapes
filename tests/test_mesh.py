#!/usr/bin/env python
"""Test mesh object and its mask."""
import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh


def test_mask() -> None:

    # with out object
    mesh = Mesh(Box[0:1, 0:1], None, [0.1, 0.1])

    target = mesh.t_mask[0]

    assert_close(mesh.t_mask[:, 0], target)
    assert_close(mesh.t_mask[:, -1], target)
    assert_close(mesh.t_mask[-1, :], target)

    # with out object
    mesh_ob = Mesh(Box[0:1, 0:1], [Box[0.3:0.6, 0.0:0.6]], [0.1, 0.1])

    target = torch.zeros_like(mesh.t_mask[0])

    t1 = target
    t1[:7] = True
    t1[-1] = True
    assert_close(mesh_ob.t_mask[3, :], t1)


@pytest.mark.parametrize(
    "domain",
    [
        (Box[0:1], None, [0.1]),
        (Box[0:1, 0:1], None, [0.1, 0.2]),
        (Box[0:1, 0:1, 0:1], None, [0.1, 0.2, 0.3]),
    ],
)
def test_mesh(domain: tuple) -> None:
    """Test field boundaries."""

    mesh = Mesh(*domain, "cpu", "double")  # type: ignore

    if mesh.dim == 1:
        assert pytest.approx(mesh.A["xl"][0]) == 0.1 * 0.1
        assert pytest.approx(mesh.V[0]) == 0.1**3
    elif mesh.dim == 2:
        assert pytest.approx(mesh.A["xl"][0]) == 0.1 * 0.2
        assert pytest.approx(mesh.A["yl"][0]) == 0.1 * 0.1
        assert pytest.approx(mesh.V[0, 0]) == 0.1**2 * 0.2
    else:
        assert pytest.approx(mesh.A["xl"][0]) == 0.2 * 0.3
        assert pytest.approx(mesh.A["yl"][0]) == 0.1 * 0.3
        assert pytest.approx(mesh.A["zl"][0]) == 0.1 * 0.2
        assert pytest.approx(mesh.V[0, 0, 0]) == 0.1 * 0.2 * 0.3
