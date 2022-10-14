#!/usr/bin/env python3
import copy
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyABC.core.boundaries import BC_TYPE_FACTORY
from pyABC.core.mesh import field_patch_mask
from pyABC.core.mesh import Mesh


@dataclass
class Field:
    """Field variable class.

    >>> var = Field(...)

    Args:
        name: name of variable.
        dim: dimension of variable. 0 to be a scalar.
        mesh: Mesh object.
        bc_config: dictionary contains boundary conditions.
        init_val: if it is given, Field will be homogeneously initialize with this valeu.
        object_interp: if True, interpolate inside of object using the boundary value of the object.

    """

    name: str
    dim: int
    mesh: Mesh
    bc_config: Optional[
        dict[str, Optional[list[dict[str, Union[float, str]]]]]
    ]
    init_val: Optional[Union[int, float]] = None
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
            elif isinstance(self.init_val, Tensor):
                for d in range(self.dim):
                    self.VAR[d, :] += self.init_val[d]
            else:
                raise ValueError("Field: unsupported data type!")

        self.set_bcs_and_masks()

    @property
    def dx(self) -> Tensor:

        return self.mesh.dx

    @property
    def nx(self) -> Tensor:

        return self.mesh.nx

    @property
    def VAR(self) -> Tensor:
        """Return working variable."""
        return self._VAR

    @VAR.setter
    def VAR(self, other: Tensor) -> None:
        self._VAR = other

    def copy(self):
        """Copy entire object."""

        return copy.deepcopy(self)

    @property
    def size(self) -> torch.Size:
        """Return self.VAR size. Return time is torch.Size which is subclass of
        `tuple`."""

        return self.VAR.size()

    def sum(self, dim: int = 0) -> Tensor:
        """Sum variable.

        Args:
            dim: dimesnion of the tensor to apply the sum operation. Defaults to 0.
        """

        return torch.sum(self.VAR, dim=dim)

    def set_var_tensor(
        self, val: Tensor, insert: Optional[int] = None
    ) -> None:
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

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:

        if isinstance(idx, slice):
            return self.VAR
        else:
            return self.VAR[idx]

    def __call__(self) -> Tensor:
        """Return variable."""

        return self.VAR

    def __add__(self, other: Any) -> Any:

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

            raise TypeError("Field: you can only add Field!")

        return self.copy()

    def __sub__(self, other: Any) -> Any:

        if isinstance(other, Field):
            self.VAR -= other()
        else:
            raise TypeError("Field: you can only subtract Field!")

        return self.copy()

    def __mul__(self, other: Any) -> Any:

        if isinstance(other, Field):
            self.VAR *= other()
        else:
            raise TypeError("Field: you can only multiply Field!")

        return self.copy()

    def __truediv__(self, other: Any) -> Any:

        if isinstance(other, Field):
            mask = other().gt(0.0)

            self.VAR[mask] /= other()[mask]
        else:
            raise TypeError("Field: you can only divide Field!")

        return self.copy()

    def set_bcs_and_masks(self) -> None:
        """Setting BCs from the given configurations.
        If there is no `Mesh.config.objs`, it will set `bcs` and `masks` to
        None.
        """

        self.bcs = []
        self.masks = dict()
        self.mask_inner = dict()

        # Setting boundary objects
        if self.bc_config is not None:

            # First domain
            if self.bc_config["domain"] is not None:
                for bc, obj in zip(
                    self.bc_config["domain"], self.mesh.domain.config
                ):

                    # Need a way to syncronize the bcs and mask!
                    self.bcs.append(
                        BC_TYPE_FACTORY[str(bc["bc_type"])](
                            "domain",
                            bc_id=obj["name"],
                            bc_obj=bc["bc_obj"],
                            bc_val=bc["bc_val"],
                            bc_var_name=self.name,
                            bc_face=obj["geometry"]["face"],
                            dtype=self.mesh.dtype,
                            device=self.mesh.device,
                        )
                    )

            if (
                self.mesh.obstacle is not None
                and self.bc_config["obstacle"] is not None
            ):
                raise NotImplementedError(
                    "Field: inner object is not supported yet!"
                )
        else:
            self.bcs_obj_set = None

        self.masks, self.masks_obj = field_patch_mask(self.mesh)

        if self.mesh.obstacle is not None and self.object_interp:
            raise NotImplementedError(
                "Field: inner obstacle is not implemented yet!"
            )
