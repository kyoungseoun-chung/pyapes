#!/usr/bin/env python
"""Test mesh object and its mask."""
from math import pi

import pytest
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.geometry import Cylinder
from pyapes.core.mesh import Mesh


def test_basic_mask() -> None:
    # With out object and for the cartesian coordinate system
    mesh = Mesh(Box[0:1, 0:1], None, [0.1, 0.1])

    assert_close(mesh.dg[0][0].mean(), mesh.dx[0] / 2)

    target = mesh.t_mask[0]

    assert_close(mesh.t_mask[:, 0], target)
    assert_close(mesh.t_mask[:, -1], target)
    assert_close(mesh.t_mask[-1, :], target)

    assert mesh.coord_sys == "xyz"

    # With object
    mesh_ob = Mesh(Box[0:1, 0:1], [Box[0.3:0.6, 0.0:0.6]], [0.1, 0.1])

    target = torch.zeros_like(mesh.t_mask[0])

    t1 = target
    t1[:7] = True
    t1[-1] = True
    assert_close(mesh_ob.t_mask[3, :], t1)

    # Test in cylindrical coordinate system
    mesh = Mesh(Cylinder[0:1, 0:1], None, [0.1, 0.1])

    assert mesh.coord_sys == "rz"


def test_geometries() -> None:
    boxes = [Box[0:2], Box[0:2, 0:2], Box[0:2, 0:2, 0:2]]

    for idx, box in enumerate(boxes):
        assert box.type == "box"
        assert box.dim == idx + 1
        assert box.size == pytest.approx(2 ** (idx + 1))
        assert box.lower == [0 for _ in range(idx + 1)]
        assert box.upper == [2 for _ in range(idx + 1)]

    cylinder = Cylinder[0:1, 0:2]

    assert cylinder.type == "cylinder"
    assert cylinder.dim == 2
    assert cylinder.size == pytest.approx(pi * 2)
    assert cylinder.lower == [0, 0]
    assert cylinder.upper == [1, 2]
