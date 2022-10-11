#!/usr/bin/env python3
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Union
from pyABC.core.mesh import Mesh, field_patch_mask

import torch
from torch import Tensor


@dataclass
class Field:

    name: str
    dim: int
    mesh: Mesh
    bc_config: Optional[dict[str, dict[str, float]]]
    init_val: Optional[Union[int, float]]
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
    def VAR(self) -> Tensor:
        """Return working variable."""
        return self._VAR

    @VAR.setter
    def VAR(self, other: Tensor) -> None:
        self._VAR = other

    def copy(self):
        """Copy entire object."""

        return copy.deepcopy(self)

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

    def add_scalar(self, idx: int, new_val: float) -> None:

        if isinstance(idx, slice):
            self.VAR += new_val
        else:
            self.VAR[idx] += new_val

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
            for bc, obj in zip(
                self.bc_config["domain"], self.mesh.domain.config
            ):

                # Need a way to syncronize the bcs and mask!
                self.bcs.append(
                    FIELD_BC_TYPE_FACTORY[bc["bc_type"]](
                        "domain",
                        bc_id=obj["name"],
                        bc_obj=bc["bc_obj"],
                        bc_val=bc["bc_val"],
                        bc_var_name=self.v_name,
                        bc_face=obj["geometry"]["face"],
                        dtype=self.mesh.dtype,
                        device=self.mesh.device,
                    )
                )

            if self.mesh.obstacle is not None:
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
