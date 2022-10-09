#!/usr/bin/env python3
"""Setup for BOX_333 mesh.

Note:
    - Also be used for the valdiation due to its geometrical simplicity
"""
import numpy as np
import torch

from pyABC.tools.utils import Container

LX_min = (0.0, 0.0, 0.0)
LX_max = (3.0, 3.0, 3.0)
DX = (0.04, 0.04, 0.04)
ROOM = "BOX 333"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def configs() -> Container:

    ##########################################################################
    # Set reference values
    ##########################################################################

    ref_configs = []

    L_ref = 1.2
    U_ref = 0.1 / 0.9
    T_ref = 295.0
    Y_ref = 1.0
    rho_ref = 1.197
    P_ref = 1e5
    nu_ref = 1.5e-5
    cp_ref = 1007.0
    g = 9.81
    lambda_ref = 0.0258

    # Reference length scale [m]
    ref_configs.append({"name": "L_ref", "val": L_ref})

    # Reference velocity [m/s]: flow rate -> 0.1*3600 over area 0.9 m^2
    ref_configs.append({"name": "U_ref", "val": U_ref})

    # Reference temperature [K]
    ref_configs.append({"name": "T_ref", "val": T_ref})

    # Reference concentration [-]
    ref_configs.append({"name": "Y_ref", "val": Y_ref})

    # Reference density [kg/m^3]
    ref_configs.append({"name": "rho_ref", "val": rho_ref})

    # Reference pressure [Pa]
    ref_configs.append({"name": "P_ref", "val": P_ref})

    # Reference kinematic viscosity [m^2/s]
    ref_configs.append({"name": "nu_ref", "val": nu_ref})

    # Reference heat capacity [J/kg*K]
    ref_configs.append({"name": "cp_ref", "val": cp_ref})

    # Gravitational acceleration [m/s^2]
    ref_configs.append({"name": "g", "val": g})

    # Lambda air
    ref_configs.append({"name": "lambda_ref", "val": lambda_ref})

    # Reference Reynolds number
    ref_configs.append(
        {"name": "Re_ref", "val": 200 * 5 / 9 * 1.2 / 1.5 / 1.2}
    )

    # Reference Prandtl number
    ref_configs.append({"name": "Pr_ref", "val": 2.0})

    # Reference Schmidt number
    ref_configs.append({"name": "Sc_ref", "val": 1.0})

    # Reference Rayleigh number
    Ra = (
        L_ref**3 * g * rho_ref * cp_ref * T_ref / nu_ref / lambda_ref / T_ref
    )
    ref_configs.append({"name": "Ra_ref", "val": Ra})

    # Reference Froude number
    Fr = U_ref / np.sqrt(g * L_ref)
    ref_configs.append({"name": "Fr_ref", "val": Fr})

    ##########################################################################
    # Objects
    ##########################################################################

    obj_configs = []

    ##########################################################################
    # Walls
    # Z is gravitational direction
    ##########################################################################

    vent_height = 0.3  # [m]

    wall_name = [
        "back_wall",
        "front_wall",
        "left_wall",
        "right_wall",
        "bottom_wall",
        "top_wall",
    ]

    # Wall normal vector
    wall_norm = [
        np.asarray([0, -1, 0], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
        np.asarray([-1, 0, 0], dtype=np.int64),
        np.asarray([1, 0, 0], dtype=np.int64),
        np.asarray([0, 0, -1], dtype=np.int64),
        np.asarray([0, 0, 1], dtype=np.int64),
    ]

    LX = np.asarray(LX_max, dtype=np.float64) - np.asarray(
        LX_min, dtype=np.float64
    )

    # Wall geometry
    wall_e_x = [
        [LX[0], 0.0, LX[2]],
        [LX[0], 0.0, LX[2]],
        [0.0, LX[1], LX[2] - vent_height],
        [0.0, LX[1], LX[2] - vent_height],
        [LX[0], LX[1], 0.0],
        [LX[0], LX[1], 0.0],
    ]

    # Wall location
    wall_x_p = [
        [0.0, 0.0, 0.0],
        [0.0, LX[1], 0.0],
        [0.0, 0.0, vent_height],
        [LX[0], 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, LX[2]],
    ]

    for name, ex, xp, dir in zip(wall_name, wall_e_x, wall_x_p, wall_norm):

        obj_configs.append(
            {
                "name": name,
                "type": "patch",
                "geometry": {"e_x": ex, "x_p": xp, "n_vec": dir},
            }
        )

    ##########################################################################
    # Inlet
    ##########################################################################
    obj_configs.append(
        {
            "name": "inlet",
            "type": "patch",
            "geometry": {
                "e_x": [0.0, LX[1], vent_height],
                "x_p": [0.0, 0.0, 0.0],
                "n_vec": [-1.0, 0.0, 0.0],
            },
        }
    )

    ##########################################################################
    # Outlet
    ##########################################################################
    obj_configs.append(
        {
            "name": "outlet",
            "type": "patch",
            "geometry": {
                "e_x": [0.0, LX[1], vent_height],
                "x_p": [LX[0], 0.0, LX[2] - vent_height],
                "n_vec": [1.0, 0.0, 0.0],
            },
        }
    )

    ##########################################################################
    # Lecturer: Currently diactivated
    # Not decided how to treat solid inside
    ##########################################################################
    # obj_configs.append(
    #     {
    #         "name": "lecturer",
    #         "type": "person",
    #         "geometry": {"e_x": [0.15, 0.4, 1.2], "x_p": [1.425, 1.3, 0.1]},
    #         "opts": {"active": True, "is_lecturer": True},
    #     }
    # )

    ##########################################################################
    # Set variables
    # id: 0-5: wall
    # id: 6 - inlet, id: 7 - outlet
    ##########################################################################
    var_configs = []

    # Velocity
    var_configs.append(
        {
            "name": "U",
            "type": "Vector",
            "dim": 3,
            "bcs": {
                # Number of bcs has to be mathed with a number of
                # object types
                # All boundary conditions need id matched with object number
                # "id": ["bc_obj_type", "bc_type", bc_val]
                "0": ["patch", "dirichlet", [0, 0, 0]],
                "1": ["patch", "dirichlet", [0, 0, 0]],
                "2": ["patch", "dirichlet", [0, 0, 0]],
                "3": ["patch", "dirichlet", [0, 0, 0]],
                "4": ["patch", "dirichlet", [0, 0, 0]],
                "5": ["patch", "dirichlet", [0, 0, 0]],
                "6": ["patch", "neumann", [0.1, 0, 0]],  # inlet
                "7": ["patch", "neumann", [0, 0, 0]]  # outlet
                # "person": {"0": ["dirichlet", [0, 0, 0]]},
            },
        }
    )

    # Pressure
    var_configs.append(
        {
            "name": "P",
            "type": "Scalar",
            "dim": 1,
            "bcs": {
                "0": ["patch", "neumann", [0]],
                "1": ["patch", "neumann", [0]],
                "2": ["patch", "neumann", [0]],
                "3": ["patch", "neumann", [0]],
                "4": ["patch", "neumann", [0]],
                "5": ["patch", "neumann", [0]],
                "6": ["patch", "neumann", [0]],
                "7": ["patch", "neumann", [0]],
                # "person": {
                #     "0": ["neumann", [0]],
                # },
            },
        }
    )

    # Temperature
    var_configs.append(
        {
            "name": "T",
            "type": "Scalar",
            "dim": 1,
            "bcs": {
                "0": ["patch", "neumann", [0]],
                "1": ["patch", "neumann", [0]],
                "2": ["patch", "neumann", [0]],
                "3": ["patch", "neumann", [0]],
                "4": ["patch", "neumann", [0]],
                "5": ["patch", "neumann", [0]],
                "6": ["patch", "neumann", [0]],
                "7": ["patch", "neumann", [0]],
                # "person": {
                #     "0": ["neumann", [0]],
                # },
            },
        }
    )

    # Scalar
    var_configs.append(
        {
            "name": "Y",
            "type": "Scalar",
            "dim": 1,
            "bcs": {
                "0": ["patch", "dirichlet", [0]],
                "1": ["patch", "dirichlet", [0]],
                "2": ["patch", "dirichlet", [0]],
                "3": ["patch", "dirichlet", [0]],
                "4": ["patch", "dirichlet", [0]],
                "5": ["patch", "dirichlet", [0]],
                "6": ["patch", "neumann", [0]],
                "7": ["patch", "neumann", [0]],
                # "person": {
                #     "0": ["dirichlet", [0]],
                # },
            },
        }
    )

    return Container(
        NAME=ROOM,
        LX_min=LX_min,
        LX_max=LX_max,
        DX=DX,
        OBJECTS=obj_configs,
        VARIABLES=var_configs,
        REFERENCE=ref_configs,
        DEVICE=DEVICE,
    )


ROOM_CONFIGS = configs()
