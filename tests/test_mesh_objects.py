#!/usr/bin/env python3
"""Try to generate mesh data based on
https://gitlab.ethz.ch/ifd-pdf/airborne-transmission/airborne_covid19/-/blob/master/geometries/mesh_MLE12.py
"""
import numpy as np
import pytest
from rich import print as rprint

from pyABC.core.boundaries import ImmersedBody
from pyABC.core.boundaries import Patch
from pyABC.core.mesh import Mesh


def test_objects() -> None:

    lx = np.asarray([3, 3, 3], dtype=np.float64)
    dx = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
    nx = np.divide(lx, dx).astype(np.int64)

    x = []
    for i in range(3):
        x.append(dx[i] * (np.arange(nx[i] + 2) - 0.5))

    test_obj = {
        "name": "left_wall",
        "type": "patch",
        "geometry": {
            "e_x": [0.0, lx[1], lx[2]],
            "x_p": [0.0, 0.0, 0.0],
            "n_vec": [-1.0, 0.0, 0.0],
        },
    }

    solid = Patch(test_obj, 0)

    from pyABC.tools.errors import FeatureNotImplementedError

    with pytest.raises(FeatureNotImplementedError) as e:
        ImmersedBody(test_obj, 0)  # <<< NOT IMPLEMENTED
        rprint(e)

    rprint(solid)


def test_with_BOX_333(report: bool = True) -> Mesh:

    from pyABC.class_rooms.BOX_333 import ROOM_CONFIGS

    mesh = Mesh(
        ROOM_CONFIGS,
    )

    if report:
        rprint(ROOM_CONFIGS)
        rprint(mesh)
        rprint(mesh.objs)
        rprint(mesh.vars)
        rprint(mesh.get("U").VAR.shape)

    return mesh


if __name__ == "__main__":
    test_objects()
    test_with_BOX_333()
