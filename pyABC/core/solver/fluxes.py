#!/usr/bin/env python3
import copy
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from pyABC.core.geometry.basis import FaceDir
from pyABC.core.geometry.basis import NormalDir

# Cubic cell
N_FACE = 6


class FluxO:
    """Flux object.

    Note:
        for the velocity, each velocity component has to be treated separately.

    Args:
        nx: number of grids
        dx: size of grids
    """

    def __init__(self, nx: npt.NDArray[np.int64], dx: npt.NDArray[np.float64]):

        self.nx = nx
        self.dx = dx

        self.shape = (nx[2], nx[1], nx[0])

        self.vol = np.prod(dx)

        self.surface = []
        self.normal = []

        # We have 6 faces due to the cubic cell
        # Fill torch.zeros to the self.faces list.
        self.faces = []

        self.schemes = []

        for dir in FaceDir:

            self.normal.append(NormalDir(dir.name))

            # Calculate cell surface area.
            # Cell is cube
            if dir.name == "w" or dir.name == "e":
                self.surface.append(dx[1] * dx[2])
            elif dir.name == "n" or dir.name == "s":
                self.surface.append(dx[0] * dx[1])
            else:
                self.surface.append(dx[0] * dx[2])

            # Initialize flux as zero
            self.faces.append(torch.zeros(self.shape, dtype=torch.float64))

        self._x = torch.zeros(self.shape, dtype=torch.float64)
        self._y = torch.zeros_like(self._x)
        self._z = torch.zeros_like(self._x)

    def __add__(self, f: Any) -> Any:  # self typing is not yet supported.
        """Add flux."""

        self.x += f.x
        self.y += f.y
        self.z += f.z

        return self

    def __mul__(self, coeff: float) -> Any:

        for f in self.faces:

            f *= coeff

        return self

    def __neg__(self) -> Any:

        for f in self.faces:
            f *= -1.0

        return self

    def numpy(self, attr: str) -> np.ndarray:

        return getattr(self, attr).detach().cpu().numpy()

    def copy(self) -> Any:
        """Copy of the object."""

        return copy.deepcopy(self)

    def copy_reset(self) -> Any:
        """Set x, y, z as zero and return deepcopy of the object."""

        copied = copy.deepcopy(self)

        copied.x = torch.zeros_like(self.x)
        copied.y = torch.zeros_like(self.y)
        copied.z = torch.zeros_like(self.z)

        return copied

    def update_scheme(self, scheme: str) -> None:

        self.schemes.append(scheme)

    def sum(self) -> None:
        """Integrate all fluxes."""

        self._c = self.x + self.y + self.z

    def set_vec(self) -> None:
        r"""Caculate x, y, z vector component.

        .. math::

            \nabla_i \Phi = \frac{1}{V_C}
            \left(f_i^+ \vec{n}^+ S^+ + f_i^- \vec{n}^- S^- \right)

        """

        self._x = self.faces[1] + self.faces[0]

        self._y = self.faces[5] + self.faces[4]

        self._z = self.faces[3] + self.faces[2]

    @property
    def c(self) -> torch.Tensor:

        return self._c

    @property
    def x(self) -> torch.Tensor:
        """X-directional flux."""

        return self._x

    @property
    def y(self) -> torch.Tensor:
        """Y-directional flux."""

        return self._y

    @property
    def z(self) -> torch.Tensor:
        """Y-directional flux."""

        return self._z

    # Assign x, y, z properties
    @x.setter
    def x(self, flux: torch.Tensor):
        self._x = flux

    @y.setter
    def y(self, flux: torch.Tensor):
        self._y = flux

    @z.setter
    def z(self, flux: torch.Tensor):
        self._z = flux

    # For the discretization. Use property method for better acecssibility
    # Get fluxes
    @property
    def w(self) -> torch.Tensor:
        """West side flux"""
        return self.faces[0]

    @w.setter
    def w(self, face: torch.Tensor):
        self.faces[0] = face

    @property
    def e(self) -> torch.Tensor:
        """East side flux"""
        return self.faces[1]

    @e.setter
    def e(self, face: torch.Tensor):
        self.faces[1] = face

    @property
    def n(self) -> torch.Tensor:
        """North side flux"""
        return self.faces[2]

    @n.setter
    def n(self, face: torch.Tensor):
        self.faces[2] = face

    @property
    def s(self) -> torch.Tensor:
        """South side flux"""
        return self.faces[3]

    @s.setter
    def s(self, face: torch.Tensor):
        self.faces[3] = face

    @property
    def f(self) -> torch.Tensor:
        """Front side flux"""
        return self.faces[4]

    @f.setter
    def f(self, face: torch.Tensor):
        self.faces[4] = face

    @property
    def b(self) -> torch.Tensor:
        """Back side flux"""
        return self.faces[5]

    @b.setter
    def b(self, face: torch.Tensor):
        self.faces[5] = face
