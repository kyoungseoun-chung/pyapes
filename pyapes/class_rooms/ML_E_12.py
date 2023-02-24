#!/usr/bin/env python3
"""Class room setup for ML E 12."""
from pyABC.tools.utils import Container

# Global parameters
LX = (13.27, 12.255, 5.53)
DX = (0.07, 0.07, 0.07)
ROOM = "ML E 12"


def configs() -> Container:
    stair_depth = 0.9
    stair_width = 9.55
    stair_height = 0.14

    obj_configs = []

    ##########################################################################
    # Stairs
    ##########################################################################

    # Stairs 0: different depth comes with
    obj_configs.append(
        {
            "name": "stair_0",
            "type": "solid",
            "geometry": {"e_x": [1.35, 9.55, 3.22], "x_p": [0.0, 0.0, 0.0]},
        }
    )

    # Stairs 1 - 5
    for i in range(5):
        height_prev = obj_configs[i]["geometry"]["e_x"][2]
        x_p0_prev = obj_configs[i]["geometry"]["x_p"][0]
        e_x0_prev = obj_configs[i]["geometry"]["e_x"][0]

        obj_name = "stair_" + str(i + 1)

        obj_configs.append(
            {
                "name": obj_name,
                "type": "solid",
                "geometry": {
                    "e_x": [
                        stair_depth,
                        stair_width,
                        height_prev - 2 * stair_height,
                    ],
                    "x_p": [x_p0_prev + e_x0_prev, 0.0, 0.0],
                },
            }
        )

    # Stairs 6
    obj_configs.append(
        {
            "name": "stair_6",
            "type": "solid",
            "geometry": {"e_x": [1.50, 12.255, 1.68], "x_p": [5.85, 0.0, 0.0]},
        }
    )

    # Stairs 7
    obj_configs.append(
        {
            "name": "stair_7",
            "type": "solid",
            "geometry": {"e_x": [0.90, 12.255, 1.54], "x_p": [7.35, 0.0, 0.0]},
        }
    )

    # Stairs 8
    obj_configs.append(
        {
            "name": "stair_8",
            "type": "solid",
            "geometry": {"e_x": [5.02, 12.255, 1.40], "x_p": [8.25, 0.0, 0.0]},
        }
    )

    idx_last_obj = len(obj_configs)
    ##########################################################################
    # Stages
    ##########################################################################
    for i in range(11):
        if i == 0:
            e_x2_prev = obj_configs[6]["geometry"]["e_x"][2]
            x_p0_prev = obj_configs[6]["geometry"]["x_p"][0]
        else:
            e_x2_prev = obj_configs[idx_last_obj + i - 1]["geometry"]["e_x"][2]
            x_p0_prev = obj_configs[idx_last_obj + i - 1]["geometry"]["x_p"][0]

        obj_name = "stage_" + str(i)
        obj_configs.append(
            {
                "name": obj_name,
                "type": "solid",
                "geometry": {
                    "e_x": [
                        0.2,
                        LX[1] - stair_width,
                        e_x2_prev - stair_height,
                    ],
                    "x_p": [x_p0_prev - 0.2, stair_width, 0],
                },
            }
        )

    idx_last_obj = len(obj_configs)
    ##########################################################################
    # Tables
    ##########################################################################
    table_depth = 0.42
    table_thickness = 0.02
    table_height = 0.28
    table_offset_from_floor = 0.76

    return Container(NAME=ROOM, LX=LX, DX=DX, OBJECTS=obj_configs)


ROOM_CONFIGS = configs()
