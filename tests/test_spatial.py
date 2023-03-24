#!/usr/bin/env python3
"""Test spatial module"""
import torch
from torch.testing import assert_close

from pyapes.core.geometry import Box
from pyapes.core.geometry import Cylinder
from pyapes.core.mesh import Mesh
from pyapes.core.variables import Field
from pyapes.tools.spatial import DiffFlux
from pyapes.tools.spatial import ScalarOP


def test_diff_flux() -> None:
    # Test cartesian
    mesh = Mesh(Box[0:1, 0:1, 0:1], None, [3, 3, 3])
    var = Field("test", 1, mesh, {"domain": None, "obstacle": None})
    var.set_var_tensor(mesh.grid[0] ** 2 + 2 * mesh.grid[2] ** 2)

    hess = ScalarOP.hess(var)

    # Test axisymmetric


def test_jac_and_hess() -> None:
    mesh = Mesh(Box[0:1, 0:1, 0:1], None, [3, 3, 3])
    var = Field("test", 1, mesh, {"domain": None, "obstacle": None})
    var.set_var_tensor(mesh.grid[0] ** 2 + 2 * mesh.grid[2] ** 2)

    jac = ScalarOP.jac(var)
    assert_close(jac.x, 2 * mesh.grid[0])
    assert_close(jac.y, torch.zeros_like(var()[0]))
    assert_close(jac.z, 4 * mesh.grid[2])

    var.set_var_tensor((mesh.grid[0] ** 2) * (mesh.grid[2] ** 2))

    hess = ScalarOP.hess(var)
    assert_close(hess.xx, 2 * mesh.grid[2] ** 2)
    assert_close(hess.xy, torch.zeros_like(var()[0]))
    assert_close(hess.xz, 4 * mesh.grid[0] * mesh.grid[2])


def test_derivative_data_structure() -> None:
    from pyapes.tools.spatial import Jac, Hess

    x = torch.rand(10)
    y = torch.rand(10)
    z = torch.rand(10)

    test_jac = Jac(x=x)

    assert len(test_jac) == 1
    assert test_jac.keys == ["x"]

    test_jac = Jac(x=x, y=y, z=z)

    assert len(test_jac) == 3
    assert test_jac.keys.sort() == ["x", "y", "z"].sort()

    for test, target in zip(test_jac, [x, y, z]):
        assert_close(test, target)

    test_jac = Jac(r=x, z=y)

    assert len(test_jac) == 2
    assert test_jac.keys.sort() == ["r", "z"].sort()

    for test, target in zip(test_jac, [x, y]):
        assert_close(test, target)

    # Test with non consecutive variable initialization
    test_hess = Hess(xx=x, yy=y)

    assert len(test_hess) == 2

    for test, target in zip(test_hess, [x, y]):
        assert_close(test, target)

    # All initialization
    test_hess = Hess(xx=x, xy=x, xz=x, yy=y, yz=y, zz=z)

    for test, target in zip(test_hess, [x, x, x, y, y, z]):
        assert_close(test, target)

    # RZ hess
    test_hess = Hess(rr=x, zz=z)
    assert test_hess.keys.sort() == ["rr", "zz"].sort()

    for test, target in zip(test_hess, [x, z]):
        assert_close(test, target)
