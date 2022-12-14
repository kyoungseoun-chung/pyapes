#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyapes.core.mesh import Mesh
from pyapes.core.variables.bcs import BC_config_type
from pyapes.core.variables.bcs import BC_FACTORY
from pyapes.core.variables.bcs import BC_type
from pyapes.core.variables.bcs import BC_val_type


@dataclass
class Field:
    """Field variable class.

    >>> var = Field(...)
    >>> var() # To get `Tensor` value of the Field

    Args:
        name: name of variable.
        dim: dimension of variable. 0 to be a scalar.
        mesh: Mesh object.
        bc_config: dictionary contains boundary conditions.
        init_val: if it is given, Field will be homogeneously initialize with this value.
        object_interp: if True, interpolate inside of object using the boundary value of the object.

    """

    name: str
    dim: int
    """Variable dimension. e.g. Scalar field has dim=1, 3D Vector field has dim=3, etc.
    Warning! This is not the same as the dimension of the mesh!"""
    mesh: Mesh
    bc_config: dict[str, list[BC_config_type] | None] | None
    init_val: int | float | list[float] | list[
        int
    ] | Tensor | str | None = None
    object_interp: bool = False

    def __post_init__(self):

        self._VAR = torch.zeros(
            self.dim,
            *self.mesh.nx,
            dtype=self.mesh.dtype.float,
            device=self.mesh.device,
            requires_grad=False,
        )

        # Initialization value
        if self.init_val is not None:

            if isinstance(self.init_val, float):
                self.VAR += self.init_val
            elif isinstance(self.init_val, list):
                assert self.dim == len(
                    self.init_val
                ), "Field: init_val should match with Field dimension!"
                for d in range(self.dim):
                    self.VAR[d] += float(self.init_val[d])
            elif isinstance(self.init_val, Tensor):
                assert self.dim == self.init_val.size(
                    0
                ), "Field: init_val should match with Field dimension!"
                for d in range(self.dim):
                    self.VAR[d] += self.init_val[d]
            elif (
                isinstance(self.init_val, str)
                and self.init_val.lower() == "random"
            ):
                self.VAR = torch.rand_like(self.VAR)
            else:
                raise ValueError("Field: unsupported data type!")

        if self.bc_config is not None:
            if "domain" not in self.bc_config:
                raise ValueError("Field: domain must be defined!")

            if "obstacle" not in self.bc_config:
                self.bc_config["obstacle"] = None

        self.set_bcs()

    @property
    def mesh_axis(self) -> list[int]:

        return [i + 1 for i in range(self.mesh.dim)]

    def set_dt(self, dt: float) -> None:
        """Set time step. Explicitly set dt for clarity."""
        self._dt = dt

    def save_old(self) -> None:
        """Save old value to `VARo`."""
        self._VARo = self.VAR.clone()

    @property
    def VARo(self) -> Tensor:
        return self._VARo

    @VARo.setter
    def VARo(self, other: Tensor) -> None:
        self._VARo = other

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dx(self) -> Tensor:
        """Mesh spacing."""

        return self.mesh.dx

    @property
    def nx(self) -> torch.Size:
        """Number of grid points."""

        return self.mesh.nx

    @property
    def VAR(self) -> Tensor:
        """Return working variable."""
        return self._VAR

    @VAR.setter
    def VAR(self, other: Tensor) -> None:
        self._VAR = other

    def copy(self, name: str | None = None) -> Field:
        """Copy entire object."""

        copied = copy.deepcopy(self)

        if name is not None:
            copied.name = name

        return copied

    def zeros_like(self, name: str | None = None) -> Field:
        """Deep copy buy init all values to zero."""

        copied = copy.deepcopy(self)
        copied._VAR = torch.zeros_like(self.VAR)

        if name is not None:
            copied.name = name

        return copied

    @property
    def size(self) -> torch.Size:
        """Return self.VAR size. Return type is torch.Size which is subclass of
        `tuple`."""

        return self.VAR.size()

    def sum(self, dim: int = 0) -> Tensor:
        """Sum variable.

        Args:
            dim: dimension of the tensor to apply the sum operation. Defaults to 0.
        """

        return torch.sum(self.VAR, dim=dim)

    def set_var_tensor(self, val: Tensor, insert: int | None = None) -> None:
        """Set variable with a given Tensor.

        Examples:
            >>> field = Field(...)
            >>> field.set_var_tensor(torch.rand(10, 10))

        Args:
            val: given values to be assigned.
            insert: inserting index. If this is specified, val is inserted
                    at `val[i==insert]`.
        """

        if self.size == val.shape:
            self._VAR = val
        else:
            for i in range(self.dim):
                if insert is not None:
                    if i == insert:
                        self._VAR[i] = val
                else:
                    self._VAR[i] = val

    def __getitem__(self, idx: int | slice) -> torch.Tensor:

        if isinstance(idx, slice):
            return self.VAR
        else:
            return self.VAR[idx]

    def __call__(self) -> Tensor:
        """Return variable."""

        return self.VAR

    def __add__(self, other: Any) -> Field:

        if isinstance(other, Field):
            self.VAR += other()
        elif isinstance(other, float):
            self.VAR += other
        elif isinstance(other, list):

            assert (
                len(other) == self.dim
            ), "Field: input vector should match with Field dimension!"

            for i in range(self.dim):
                self.VAR[i] += other[i]

        elif isinstance(other, Tensor):

            if other.size(0) == self.dim:
                self.VAR = other
            else:
                for i in range(other.size(0)):
                    self.VAR[i] += other[i]
        else:

            raise TypeError(
                "Field: you can only add Field, float, Tensor, list[int], or list[float]!"
            )

        return self.copy()

    def __sub__(self, other: Any) -> Field:

        if isinstance(other, Field):
            self.VAR -= other()
        else:
            raise TypeError("Field: you can only subtract Field!")

        return self.copy()

    def __mul__(self, other: Any) -> Field:

        if isinstance(other, Field):
            self.VAR *= other()
        elif isinstance(other, Union[float, int]):
            self.VAR *= other
        else:
            raise TypeError(
                "Field: you can only multiply Field, int, or float!"
            )

        return self.copy()

    def __truediv__(self, other: Any) -> Field:

        if isinstance(other, Field):
            mask = other().gt(0.0)

            self.VAR[mask] /= other()[mask]
        else:
            raise TypeError("Field: you can only divide by Field!")

        return self.copy()

    def set_bcs(self) -> None:
        """Setting BCs from the given configurations.
        If there is no `Mesh.config.objs`, it will set `bcs` and `masks` to
        None.
        """

        self.bcs: list[BC_type] = []

        # Setting boundary objects
        if self.bc_config is not None:

            # First domain
            if self.bc_config["domain"] is not None:

                d_obj_config = self.mesh.domain.config
                d_bc_config = self.bc_config["domain"]

                for bc, obj in zip(d_bc_config, d_obj_config):

                    # Not sure about a proper typing checking here...
                    bc_val = cast(BC_val_type, bc["bc_val"])
                    bc_face = cast(str, bc["bc_face"])

                    self.bcs.append(
                        BC_FACTORY[str(bc["bc_type"])](
                            bc_id=f"d-{obj}",
                            bc_val=bc_val,
                            bc_face=bc_face,
                            bc_mask=self.mesh.d_mask[bc_face],
                            bc_var_name=self.name,
                            dtype=self.mesh.dtype,
                            device=self.mesh.device,
                        )
                    )

            if (
                self.mesh.obstacle is not None
                and self.bc_config["obstacle"] is not None
            ):
                raise NotImplementedError
