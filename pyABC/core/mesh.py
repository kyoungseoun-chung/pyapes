#!/usr/bin/env python3
import time

import numpy as np
import rich.repr
import torch

from pyABC.core.fields import Variables
from pyABC.tools.errors import SizeDoesNotMatchError
from pyABC.tools.errors import WrongInputError
from pyABC.tools.progress import RichPBar
from pyABC.tools.utils import Container
from pyABC.tools.utils import getsize

DEVICE_AVAILABLE = ["cpu", "cuda"]
NDIM = 3

# Currently only support velocity, pressure, temperature, and passive scalars
VAR_TYPES = ["U", "P", "T", "Y"]


class Mesh:
    """Construct Mesh class.

    Args:
        config: configuration container

    Note:
        - The container shoud contains,
          NAME, LX, DX, OBJECTS, VARIABLES, REFERENCE, and DEVICE.

        Example:

            >>> Container(
                NAME=str, LX=tuple, DX=tuple, OBJECTS=dict,
                VARIABLES=dict, REFERENCE=dict, DEVICE=str
            )

    """

    def __init__(self, config: Container, simple: bool = False):

        # Store information of mesh icnluding boundary conditions
        self.config = config
        self.simple = simple

        # Name of the room
        self.room = config.NAME

        # Device
        self.device = config.DEVICE

        self.LX_min = self.config.LX_min
        self.LX_max = self.config.LX_max

        self.Dx = self.config.DX

        # Set device for torch.Tensor
        try:
            self.device = torch.device(self.device.lower())
        except RuntimeError:
            msg = (
                f"Unsupported device is given: {self.device}. "
                f"Should be one of {DEVICE_AVAILABLE}!"
            )
            raise WrongInputError(msg)

        self.lx = np.asarray(self.LX_max, dtype=np.float64) - np.asarray(
            self.LX_min, dtype=np.float64
        )
        self.dx = np.asarray(self.Dx, dtype=np.float64)
        # Cell volume. Assume equi-distance and cubic cell
        self.vol = np.prod(self.dx)

        self.nx = np.divide(self.lx, self.dx).astype(np.int64)

        self.x = []

        # Node is at the cell center
        # bc|- 0 -|- 1 -|- 2 -|bc
        # bc|- dx-|- dx-|- dx-|bc
        for i in range(NDIM):
            self.x.append(
                self.dx[i] * (np.arange(self.nx[i]) + 0.5) + self.LX_min[i]
            )

        # Mesh grid
        self.X3, self.X2, self.X1 = np.meshgrid(
            self.x[2], self.x[1], self.x[0], indexing="ij"
        )

        if not self.simple:
            # The flag self.simple only gives grid information.
            # Therefore, if self.simple == false, add reference, objects,
            # and variables

            self.refs = dict()
            self.objs = []
            self.vars = dict()

            # Add reference values
            self.add_references(self.config.REFERENCE)
            # Add objects inside the domain. Mask will be created right afterward
            self.add_objects(self.config.OBJECTS)
            # Add field variables
            self.add_variables(self.config.VARIABLES)

    @property
    def is_cuda(self) -> bool:
        """Check the device."""

        return self.device == torch.device("cuda")

    def add_references(self, configs: list[dict]) -> None:
        """Add reference values.

        Args:
            configs (dict): configuration of the reference values
                - Should have: name and val
        """

        for ref in configs:

            ref_name = ref["name"]
            ref_val = ref["val"]

            self.refs[ref_name] = ref_val

    def add_objects(self, configs: list[dict]) -> None:
        """Add objects inside of the Mesh.

        Args:
            configs (dict): configuration of the mesh objects
                - Should have: e_x, x_p, x, dx
                - Optional: id, active, is_lecturer, bubble
        """
        from pyABC.core.boundaries import BC_OBJECT_FACTORY

        pbar = RichPBar(configs, desc="MESH", mininterval=0)

        # Global BC object id
        id = 0
        # Add objects according to the configurations
        for obj in pbar:

            obj_config = obj

            obj_type = obj_config["type"]

            # Create object
            if obj_type in BC_OBJECT_FACTORY:

                self.objs.append(BC_OBJECT_FACTORY[obj_type](obj_config, id))
                id += 1
            else:

                msg = (
                    f"Unsupported object type! (given: {obj_type}). "
                    f"Should be one of {list(BC_OBJECT_FACTORY.keys())}"
                )
                WrongInputError(msg)

            desc_mod = f"MESH: {obj_type} | {id}"
            pbar.set_description(desc_mod)
            # Add delay for to visualize progress properly
            time.sleep(0.1)

    def add_variables(self, configs: list[dict]) -> None:
        """Add variables used in the simulation.

        Args:
            configs (dict): configuration of the variables
                - Should have: name, dim
        """

        pbar = RichPBar(configs, desc="VARIABLES:", mininterval=0)

        # Add objects according to the configurations
        for var in pbar:
            # Get object paramters
            var_name = var["name"]
            var_type = var["type"]
            var_dim = var["dim"]
            var_bcs = var["bcs"]

            # Check the number of bcs definition
            if len(var_bcs) != len(self.objs):

                msg = (
                    f"A number of bcs defined ({len(var_bcs)}) has to be same "
                    "with a number of objects types defined "
                    f"({len(self.objs)})"
                )

                raise SizeDoesNotMatchError(msg)

            self.vars[var_name] = Variables(
                var_name,
                var_type,
                var_dim,
                self.objs,
                var_bcs,
                self.x,
                self.nx,
                self.dx,
                self.device,
            )
            desc_mod = f"VARIABLES: {var_name}"
            pbar.set_description(desc_mod)
            # Add delay for to visualize progress properly
            time.sleep(0.1)

    def get(self, name: str) -> Variables:
        """Return variables and masks by name

        Note:
            - If, name is one of VAR_TYPES, return Variables.VAR accordingly

        Args:
            name (str): name of variables to be return
        """

        if name in VAR_TYPES:
            return self.vars[name]
        else:

            raise WrongInputError(f"Wrong variable name is given! {name}")

    def __rich_repr__(self) -> rich.repr.Result:

        yield f"{self.room}"
        yield "\t- Lx", self.lx
        yield "\t- dx", self.dx
        yield "\t- nx", self.nx
        yield "\t- memory", getsize(self)
