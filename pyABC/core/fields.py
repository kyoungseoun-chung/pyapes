#!/usr/bin/env python3
"""Contains field data."""
from dataclasses import dataclass
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import rich.repr
import torch


# Slice indecies
C = slice(1, -1)  # i
P = slice(2, None)  # i+1
M = slice(None, -2)  # i-1

VARIABLE_TYPES = ["vector", "scalar"]


@dataclass
class Variables:
    """Dataclass contains a field variables used in the simulation.

    Note:
        - Variables are initialized in NX size
        - Fluxes are stored and can be called by the property method
          (w, e, n, s, f, b, p => west, east, north, front, back, center)
        - Mesh consists of cubic cells
        - VAR coordinate is [dim, z, y, x]: (z: front-back. y: north-south,
          x: east-west)

    Args:
        name: user defined variable name
        type: type of a variable
            - Should be one of [velocity, pressure, temperature, scalar, and mask]
        dim: leading dimension of the variable.
            - e.g. `"U".dim = 3, "P".dim = 1`
            - Currently not sure how to treat multi-dimensional scalar field
        objs: boundary objects. One of Patch or ImmersedBody or None
        bc_config: boundary configuration.

            Note:

                If this is not provided, you can just use this class as
                variable container

        x: mesh coordinates
        NX: a number of grid points
        DX: grid spacing
        device: either torch.device("cpu") or torch.device("cuda")
    """

    name: str
    type: str
    dim: int
    objs: Optional[list]  # Why list[Patch, ImmersedBody] is not working?
    bc_config: Optional[dict]
    x: list[npt.NDArray[np.float64]]
    NX: npt.NDArray[np.int64]
    DX: npt.NDArray[np.float64]
    device: torch.device

    def __post_init__(self):

        # Make sure
        self.type = self.type.lower()

        if self.type not in VARIABLE_TYPES:
            from pyABC.tools.errors import WrongInputError

            msg = f"Unsupported variable type! (self.type)"
            raise WrongInputError(msg)

        # Set VAR. If device is "cuda", use torch.Tensor else, np.ndarray
        if self.dim > 1:
            self.VAR = torch.zeros(
                (self.dim, self.NX[2], self.NX[1], self.NX[0]),
                dtype=torch.float64,
                device=self.device,
            )
        else:
            self.VAR = torch.zeros(
                (self.NX[2], self.NX[1], self.NX[0]),
                dtype=torch.float64,
                device=self.device,
            )

        # Add boundary conditions if self.objs and self.bc_config is provided
        if self.bc_config is not None and self.objs is not None:

            from pyABC.core.boundaries import BOUNDARY_FACTORY
            from pyABC.core.boundaries import create_patch_mask

            var_bcs = []

            for bc in self.bc_config:
                # Create BC object
                bc_id = bc
                bc_obj_type = self.bc_config[bc][0]
                bc_type = self.bc_config[bc][1]

                if self.bc_config[bc][2] is None:
                    bc_val = None
                else:
                    bc_val = torch.tensor(
                        self.bc_config[bc][2], dtype=torch.float64
                    )

                var_bcs.append(
                    BOUNDARY_FACTORY[bc_type](
                        bc_obj_type,
                        bc_type,
                        bc_id,
                        bc_val,
                        self.name,
                        self.device,
                    )
                )
                # Create mask of the BC object

            self.bcs = var_bcs

            # Create BC mask
            self.masks = create_patch_mask(
                self.x, self.DX, self.NX, self.objs, self.bcs, self.device
            )
        else:
            self.bcs = None

    def __call__(self) -> torch.Tensor:
        """Return Variable's tensor."""

        return self.VAR

    def __eq__(self, var: Any) -> bool:
        """If dimension of Variables are the same, return True.
        This checks number of grids and grid spacing to make sure two
        variables are lying on the same mesh.
        """

        return np.prod(self.NX == var.NX) and np.prod(self.DX == var.DX)

    def __ne__(self, var: Any) -> bool:
        """If dimension of Variables are not the same, return True.

        Note:
            - Eiter NX is different or DX is different is different.
        """

        return np.prod(self.NX != var.NX) or np.prod(self.DX != var.DX)

    def __neg__(self) -> Any:
        """Override negative operator."""

        self.VAR = -self.VAR

        return self

    def set_var_matrix(self, val: npt.NDArray[np.float64]) -> None:
        """Set variable with a given matrix.

        Args:
            val: given values to be assigned.
        """
        if self.dim > 1:
            for i in range(self.dim):
                self.VAR[i] = torch.from_numpy(val[i]).to(self.device)
        else:
            self.VAR = torch.from_numpy(val).to(self.device)

    def set_var_vector(self, val: npt.NDArray[np.float64]) -> None:
        """Set variable with a given vector.

        Args:
            val: given values to be assigned.
        """

        for i in range(self.dim):
            self.VAR[i] = torch.ones_like(self.VAR[i]) * val[i]

    def set_var_scalar(self, val: float) -> None:
        """Set variable with a given value (scalar).

        Args:
            val: given values to be assigned.
        """

        self.VAR = torch.zeros_like(self.VAR) + val

    @property
    def save_old(self) -> None:

        self.VAR_o = self.VAR.clone()

    @property
    def is_cuda(self) -> bool:
        """Check whether we are using cuda or not."""
        return self.device == torch.device("cuda")

    @property
    def is_cpu(self) -> bool:
        """Check whether we are using cpu or not."""
        return self.device == torch.device("cpu")

    def __rich_repr__(self) -> rich.repr.Result:

        yield f"{self.type}"
        yield "\t- nx", self.NX
        yield "\t- dim", self.dim
        if self.bcs is not None:
            yield "\t- bcs", self.bcs
