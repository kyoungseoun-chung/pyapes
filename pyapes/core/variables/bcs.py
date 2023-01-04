#!/usr/bin/env python3
"""New boundary conditions for FVM method. Therefore, all boundary conditions will be applied to the faces of `.fluxes.Flux` class.

Supporting types:
    * Dirichlet
    * Neumann
    * Symmetry
    * Periodic
"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import get_args
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyapes.core.backend import DType
from pyapes.core.geometry.basis import DIR_TO_NUM
from pyapes.core.geometry.basis import FDIR
from pyapes.core.variables.fields import Field
from pyapes.core.variables.fluxes import Flux

BC_val_type = Union[
    int,
    float,
    list[int],
    list[float],
    Callable[[tuple[Tensor, ...], Tensor, int], Tensor],
]
"""BC value type."""
BC_config_type = dict[str, Union[str, BC_val_type]]
"""BC config type."""


@dataclass
class BC(ABC):
    """Base class of the boundary condition object.

    Note:
        - Only designed to be used for the scalar field not vector field.
        - Therefore, we have leading dimension of 0 on var.

    Args:
        bc_id: BC id
        bc_obj: BC object type (:py:mod:`Patch` or :py:mod:`InnerObject`)
        bc_type: BC type either (:py:mod:`Neumann` or :py:mod:`Dirichlet`)
        bc_val: values for the boundary condition. Either `list[float]` or
            `float` or `Callable`
        bc_mask: boundary mask according to the face of the mesh
        bc_var_name: name of the variable
        dtype: :py:mod:`DType`
        device: torch.device. Either `cpu` or `cuda`
    """

    bc_id: str
    bc_val: BC_val_type
    bc_face: str
    bc_mask: Tensor
    bc_var_name: str
    dtype: DType
    device: torch.device

    def __post_init__(self):

        _bc_val_type_check(self.bc_val)
        self.bc_face_dim = DIR_TO_NUM[self.bc_face[0]]

        if self.bc_face[-1] == "l":
            self.bc_sign = -1.0
        else:
            self.bc_sign = 1.0

    @abstractmethod
    def apply(self, var: Field, grid: tuple[Tensor, ...], order: int) -> None:
        """Apply boundary conditions. Combination of `self.at_bc` and `self.to_bc`."""
        ...


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions."""

    def apply(self, var: Field, grid: tuple[Tensor, ...]) -> None:
        """Apply BC"""

        assert not isinstance(
            self.bc_val, dict
        ), f"BC: {self.__class__.__name__} does not support dict type boundary value."

        for i in range(var.dim):
            if callable(self.bc_val):
                var()[i, self.bc_mask] = self.bc_val(grid, self.bc_mask, i)
            elif isinstance(self.bc_val, list):
                var()[i, self.bc_mask] = self.bc_val[i]
            else:
                var()[i, self.bc_mask] = self.bc_val


class Neumann(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(self, var: Field, grid: tuple[Tensor, ...], order=int) -> None:
        """Apply BC"""

        raise NotImplementedError


class Symmetry(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(self, var: Tensor, flux: Flux, *_) -> None:
        """Apply BC"""
        dim = var.size(0)

        for d in range(dim):

            if self.bc_face[1] == "l":
                opp_face = self.bc_face.replace("l", "r")
            else:
                opp_face = self.bc_face.replace("r", "l")

            face_val = flux.face(d, self.bc_face)
            opp_face_val = flux.face(d, opp_face)

            face_val[self.bc_mask] = opp_face_val[self.bc_mask]
            flux.to_face(d, self.bc_face[0], self.bc_face[1], face_val)


class Periodic(BC):
    r"""Apply Periodic boundary conditions."""

    def apply(self, *_) -> None:
        """Apply BC"""
        # Do nothing
        pass


def _bc_val_type_check(bc_val: BC_val_type):

    # NOTE: NOT SURE ABOUT THIS.. TYPING CHECKING IS WEIRD HERE..
    if not isinstance(bc_val, Callable):

        if type(bc_val) not in get_args(BC_val_type):
            raise TypeError(
                f"BC: wrong bc variable -> {type(bc_val)} is not one of {get_args(BC_val_type)}!"
            )


def homogeneous_bcs(
    dim: int, bc_val: Optional[float], bc_type: str
) -> list[BC_config_type]:
    """Simple pre-defined boundary condition.
    Args:
        dim: dimension of mesh
        bc_val: value at the boundary. Homogenous at the boundary surface.
        bc_type: Type of bc
    """

    bc_config = []
    for i in range(dim * 2):
        bc_config.append(
            {"bc_face": FDIR[i], "bc_type": bc_type, "bc_val": bc_val}
        )
    return bc_config


class BC_HD:
    """Homogeneous Dirichlet BC for the box. Just a useful tool."""

    def __new__(cls, dim: int, bc_val: float):
        return homogeneous_bcs(dim, bc_val, "dirichlet")


class BC_HN:
    """Homogeneous Neumann BC for the box. Just a useful tool."""

    def __new__(cls, dim: int, bc_val: float):
        return homogeneous_bcs(dim, bc_val, "neumann")


BC_type = Union[Dirichlet, Neumann, Symmetry, Periodic]


BC_FACTORY = {
    "dirichlet": Dirichlet,
    "neumann": Neumann,
    "symmetry": Symmetry,
    "periodic": Periodic,
}
