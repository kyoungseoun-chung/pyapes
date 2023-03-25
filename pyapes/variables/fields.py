#!/usr/bin/env python3
"""Module contains the `Field` class."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from typing import Union

import torch
from torch import Tensor

from pyapes.mesh import Mesh
from pyapes.variables.bcs import BC_FACTORY
from pyapes.variables.bcs import BC_type
from pyapes.variables.bcs import BCConfig


@dataclass
class Field:
    """Field variable class.

    Examples:
        >>> from pyapes.core.geometry import Box
        >>> from pyapes.core.mesh import Mesh
        >>> from pyapes.core.variables.bcs import homogeneous_bcs

        >>> mesh = Mesh(Box[0 : 2 * pi], None, [21])
        >>> f_bc = homogeneous_bcs(1, None, "periodic")
        >>> var = Field("U", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.5)

    """

    name: str
    """Name of the field variable."""
    dim: int
    """Variable dimension. e.g. Scalar field has dim=1, 3D Vector field has dim=3, etc.
    Warning! This is not the same as the dimension of the mesh!"""
    mesh: Mesh
    """Mesh object."""
    bc_config: dict[str, list[BCConfig] | None] | None
    """Boundary configuration"""
    init_val: int | float | list[float] | list[int] | Tensor | list[
        Tensor
    ] | str | None = None
    """Assign initial value to the field by `init_val`."""
    object_interp: bool = False
    """If object inside and `object_interp=True`, interpolate the inside value of the object."""

    def __post_init__(self):
        # Initialize the variable as a zero tensor
        self._VAR = torch.zeros(
            self.dim,
            *self.mesh.nx,
            dtype=self.mesh.dtype.float,
            device=self.mesh.device,
            requires_grad=False,
        )

        # NOTE: Require refactoring
        # NOTE: Also need load data from file
        # Initialization value
        if self.init_val is not None:
            if isinstance(self.init_val, float):
                self.VAR += self.init_val

            elif isinstance(self.init_val, list):
                assert self.dim == len(
                    self.init_val
                ), "Field: init_val should match with Field dimension!"

                if isinstance(self.init_val[0], float):
                    for d in range(self.dim):
                        self.VAR[d] += float(self.init_val[d])
                elif isinstance(self.init_val[0], Tensor):
                    for d in range(self.dim):
                        self.VAR[d] += self.init_val[d]
                else:
                    raise ValueError(
                        f"Field: {type(self.init_val[0])} is an unsupported init_val type!"
                    )

            elif isinstance(self.init_val, Tensor):
                assert self.dim == self.init_val.size(
                    0
                ), "Field: init_val should match with Field dimension!"
                for d in range(self.dim):
                    self.VAR[d] += self.init_val[d]

            elif isinstance(self.init_val, str) and self.init_val.lower() == "random":
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

    def set_time(self, dt: float, init_val: float | None = None) -> None:
        """Set field time stamp with `self.dt`. If `init` is provided, set time stamp to `init`."""

        if init_val is not None:
            self._t = init_val
        else:
            self._t = 0.0

        self._dt = dt

    def update_time(self, dt: float | None = None) -> None:
        """Update time step by `dt`. If `dt` is None, update by `self.dt`."""

        self._t += self.dt if dt is None else dt

    @property
    def t(self) -> float:
        """Time stamp of the field."""
        return self._t

    def save_old(self) -> None:
        """Save old value to `VARo`."""
        self._VARo = self.VAR.clone()

    @property
    def VARo(self) -> Tensor:
        """Old variable stored before the time integration."""
        return self._VARo

    @VARo.setter
    def VARo(self, other: Tensor) -> None:
        self._VARo = other

    @property
    def dt(self) -> float:
        """Time step size."""
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
        """Deep copy but init all values to zero."""

        copied = copy.deepcopy(self)
        copied._VAR = torch.zeros_like(self.VAR)

        if name is not None:
            copied.name = name

        return copied

    def zeros_like_tensor(self) -> Tensor:
        """Just return `torch.zeros_like(self.VAR)`."""
        return torch.zeros_like(self.VAR)

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

    def set_var_tensor(
        self,
        val: Tensor,
        insert: int | None = None,
    ) -> Field:
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
        return self

    def __getitem__(self, idx: int | slice) -> torch.Tensor:
        if isinstance(idx, slice):
            return self.VAR
        else:
            return self.VAR[idx]

    def __call__(self) -> Tensor:
        """Return variable."""

        return self.VAR

    def __add__(self, other: Any) -> Field:
        """Use `+` operator to add values to the field."""

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

        return self

    def __sub__(self, other: Any) -> Field:
        """Use `-` operator to subtract `other` from `Field.VAR`."""

        if isinstance(other, Field):
            self.VAR -= other()
        else:
            raise TypeError("Field: you can only subtract Field!")

        return self

    def __mul__(self, other: Any) -> Field:
        """Use `*` operator to multiply `Field.VAR` by `other`."""

        if isinstance(other, Field):
            self.VAR *= other()
        elif isinstance(other, Union[float, int]):
            self.VAR *= other
        else:
            raise TypeError("Field: you can only multiply Field, int, or float!")

        return self

    def __truediv__(self, other: Any) -> Field:
        """Use `/=` operator to divide `Field.VAR` by `other`."""

        if isinstance(other, Field):
            mask = other().gt(0.0)

            self.VAR[mask] /= other()[mask]
        else:
            raise TypeError("Field: you can only divide by Field!")

        return self

    def __ilshift__(self, other: Any) -> Field:
        """use `<<=` operator to assign `other` to `Field.VAR`."""

        if isinstance(other, Field):
            self.VAR = other()
        elif isinstance(other, Tensor):
            self.set_var_tensor(other)
        elif isinstance(other, float | int):
            self.VAR = torch.zeros_like(self.VAR) + other
        elif isinstance(other, list):
            assert self.dim == len(other), "Field: dimension mismatch!"
            self.VAR = torch.zeros_like(self.VAR)
            for i in range(self.dim):
                self.VAR[i] += other[i]
        else:
            raise TypeError(
                "Field: you can only assign Field, Tensor, float, int, or list!"
            )

        return self

    def get_bc(self, bc_id: str) -> BC_type | None:
        """Get bc object by bc_id. bc_id should have convention `d-xl` for domain boundary on `xl` side.

        If no bc is found, it will return None. If multiple bcs are found, it will raise KeyError.
        """

        bc_found = [bc for bc in self.bcs if bc.bc_id == bc_id]

        if len(bc_found) == 0:
            return None
        elif len(bc_found) > 1:
            raise KeyError(
                f"Field: bc_id {bc_id} returned multiple bcs. Check id once again!"
            )
        else:
            return bc_found[0]

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

                assert len(d_obj_config) == len(
                    d_bc_config
                ), f"Field: domain config ({len(d_obj_config)}) mismatch with bc config ({len(d_bc_config)})!"

                for bc in d_bc_config:
                    # Not sure about a proper typing checking here...
                    bc_val = bc["bc_val"]
                    bc_val_opt = bc["bc_val_opt"] if "bc_val_opt" in bc else None
                    bc_face = bc["bc_face"]

                    self.bcs.append(
                        BC_FACTORY[str(bc["bc_type"])](
                            bc_id=f"d-{bc_face}",
                            bc_val=bc_val,
                            bc_val_opt=bc_val_opt,
                            bc_face=bc_face,
                            bc_mask=self.mesh.d_mask[bc_face],
                            bc_var_name=self.name,
                            bc_coord_sys=self.mesh.coord_sys,
                            mesh_dim=self.mesh.dim,
                            dtype=self.mesh.dtype,
                            device=self.mesh.device,
                        )
                    )

            if (
                self.mesh.obstacle is not None
                and self.bc_config["obstacle"] is not None
            ):
                raise NotImplementedError
