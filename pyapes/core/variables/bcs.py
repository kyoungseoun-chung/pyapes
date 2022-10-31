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
from pyapes.core.variables.fluxes import Flux

BC_val_type = Union[int, float, list[int], list[float], Callable]
BC_config_type = dict[str, Union[str, BC_val_type]]


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

    @abstractmethod
    def apply(
        self, var: Tensor, flux: Flux, grid: tuple[Tensor], order: int
    ) -> Tensor:
        """Apply boundary conditions.

        Args:
            mask: mask of bc object
            var: Field values at the volume center
            flux: Flux object to apply the boundary condition
            grid: `Mesh.grid` to be used for `Callable` `self.bc_val`
            order: order of boundary evaluation. if `order` is zero, face value will bel linearly evaluated in between cell `i` and `i+1`. else `order` is non-zero, face value will be `order`-derivative at the face.
        """
        ...


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions."""

    def apply(
        self, var: Tensor, flux: Flux, grid: tuple[Tensor], order: int
    ) -> None:
        """Apply BC"""
        dim = var.size(0)

        # NOTE: I'd like to integrate callable too...
        for d in range(dim):

            face_val = flux.face(d, self.bc_face)

            if order == 0:
                face_val[self.bc_mask] = (
                    self.bc_val[d]
                    if isinstance(self.bc_val, list)
                    else self.bc_val(grid, self.bc_mask)[d]
                    if isinstance(self.bc_val, Callable)
                    else self.bc_val
                )
                flux.to_face(d, self.bc_face[0], self.bc_face[1], face_val)
            elif order == 1:
                dx = flux.mesh.dx
                pass
                face_val[self.bc_mask] = (
                    (self.bc_val[d] - var[d][self.bc_mask])
                    / (2 * dx[self.bc_face_dim])
                    if isinstance(self.bc_val, list)
                    else (
                        self.bc_val(grid, self.bc_mask)[d]
                        - var[d][self.bc_mask]
                    )
                    / (2 * dx[self.bc_face_dim])
                    if isinstance(self.bc_val, Callable)
                    else (self.bc_val - var[d][self.bc_mask])
                    / (2 * dx[self.bc_face_dim])
                )
            else:
                raise ValueError(
                    f"BC: boundary value evaluation for {order}-derivative is not supported!"
                )


class Neumann(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(
        self, var: Tensor, flux: Flux, grid: tuple[Tensor], order=int
    ) -> None:
        """Apply BC"""
        dim = var.size(0)

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
