#!/usr/bin/env python
"""Test mesh object and its mask."""
import torch
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh


def test_mask() -> None:

    # with out object
    mesh = Mesh(Box[0:1, 0:1], None, [0.1, 0.1])

    assert_close(mesh.dg[0][0].mean(), mesh.dx[0] / 2)

    target = mesh.t_mask[0]

    assert_close(mesh.t_mask[:, 0], target)
    assert_close(mesh.t_mask[:, -1], target)
    assert_close(mesh.t_mask[-1, :], target)

    # with object
    mesh_ob = Mesh(Box[0:1, 0:1], [Box[0.3:0.6, 0.0:0.6]], [0.1, 0.1])

    target = torch.zeros_like(mesh.t_mask[0])

    t1 = target
    t1[:7] = True
    t1[-1] = True
    assert_close(mesh_ob.t_mask[3, :], t1)
    pass
