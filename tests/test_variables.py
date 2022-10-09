#!/usr/bin/env python3
"""Test field (variable) module."""
import numpy as np
import torch
from rich import print as rprint


def grid_setup(
    l_min: list = [0, 0, 0],
    l_max: list = [1, 1, 1],
    dx: list = [0.2, 0.2, 0.2],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:

    LX = np.asarray(l_max, dtype=np.float64) - np.asarray(
        l_min, dtype=np.float64
    )
    DX = np.asarray(dx, dtype=np.float64)
    NX = np.divide(LX, DX).astype(np.int64)

    X = []
    for i in range(3):
        X.append(DX[i] * (np.arange(NX[i]) + 0.5) + l_min[i])

    return LX, DX, NX, X


def test_get_box_333(report: bool = False):
    from tests.test_mesh_objects import test_with_BOX_333

    mesh = test_with_BOX_333(report=False)

    if report:
        rprint(mesh)

    return mesh


def test_variables(report: bool = True):

    from pyABC.core.boundaries import Patch

    from pyABC.core.fields import Variables

    _, dx, nx, x = grid_setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj_config = [
        {
            "name": "bc_test_back_wall",
            "type": "patch",
            "geometry": {
                "e_x": [1.0, 0.0, 1.0],
                "x_p": [0.0, 0.0, 0.0],
                "n_vec": np.asarray([0, -1, 0], dtype=np.int64),
            },
        },
        {
            "name": "bc_test_front_wall",
            "type": "patch",
            "geometry": {
                "e_x": [1.0, 0.0, 1.0],
                "x_p": [0.0, 1.0, 0.0],
                "n_vec": np.asarray([0, 1, 0], dtype=np.int64),
            },
        },
    ]

    # Create Patch object
    objs = [Patch(obj_config[0], 0), Patch(obj_config[1], 1)]

    if report:
        rprint(objs)

    # BCs
    vel_bcs_config = {
        "0": ["patch", "dirichlet", [0, 10, 0]],
        "1": ["patch", "neumann", [0, -1, 0]],
    }

    var = dict()

    # Create Variables object to apply the boundary condition
    var["U"] = Variables(
        name="U",
        type="Vector",
        dim=3,
        objs=objs,
        bc_config=vel_bcs_config,
        x=x,
        NX=nx,
        DX=dx,
        device=device,
    )

    var["P"] = Variables(
        name="P",
        type="Scalar",
        dim=1,
        objs=objs,
        bc_config=vel_bcs_config,
        x=x,
        NX=nx,
        DX=dx,
        device=device,
    )

    var["Y"] = Variables(
        name="Y",
        type="Scalar",
        dim=1,
        objs=None,
        bc_config=None,
        x=x,
        NX=nx,
        DX=dx,
        device=device,
    )

    if report:
        # Check members of variable
        rprint(var)
        rprint(var["U"].is_cpu)
        rprint(var["U"].is_cuda)
        rprint(var["U"].masks)

        assert var["U"].VAR.device == device

        # Check value assignment
        var_shape = var["U"].VAR.shape
        p_shape = var["P"].VAR.shape
        var["U"].set_var_scalar(10.0)

        assert var["U"].VAR.shape == var_shape
        assert var["U"].VAR[0, 0, 0, 0] == 10.0

        input_vector = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        var["U"].set_var_vector(input_vector)

        assert var["U"].VAR[0, 0, 0, 0] == 1.0
        assert var["U"].VAR[1, 0, 0, 0] == 2.0
        assert var["U"].VAR[2, 0, 0, 0] == 3.0

        # Somehow, np.random.rand causes type error message here.
        input_matrix = np.asarray(
            np.random.rand(*(var_shape)), dtype=np.float64
        )
        var["U"].set_var_matrix(input_matrix)

        assert var["U"].VAR[0, 0, 0, 0] == input_matrix[0, 0, 0, 0]
        assert var["U"].VAR[1, 0, 0, 0] == input_matrix[1, 0, 0, 0]
        assert var["U"].VAR[2, 0, 0, 0] == input_matrix[2, 0, 0, 0]

        input_matrix = np.asarray(np.random.rand(*(p_shape)), dtype=np.float64)

        var["P"].set_var_matrix(input_matrix)
        assert var["P"].VAR[0, 0, 0] == input_matrix[0, 0, 0]

        rprint(var["U"]())

        assert var["P"] == var["P"]

    return var


if __name__ == "__main__":
    test_variables()
