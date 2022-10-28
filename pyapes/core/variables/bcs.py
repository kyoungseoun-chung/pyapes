#!/usr/bin/env python3
"""New boundary conditions for FVM method. Therefore, all boundary conditions will be applied to the faces of `.fluxes.Flux` class.

Supporting types:
    * Dirichlet
    * Neumann
    * Symmetry
    * Periodic
"""

import torch
from torch import Tensor


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Type, Union, get_args

from pyapes.core.backend import DType
from pyapes.core.variables.fluxes import Flux
from pyapes.core.geometry.basis import FDIR_TO_NUM

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
        bc_var_name: name of the variable
        dtype: :py:mod:`DType`
        device: torch.device. Either `cpu` or `cuda`
    """

    bc_id: str
    bc_val: BC_val_type
    bc_face: str
    bc_var_name: str
    dtype: DType
    device: torch.device

    def __post_init__(self):

        _bc_val_type_check(self.bc_val)
        self.face_n_dir = -1 if FDIR_TO_NUM[self.bc_face] % 2 == 0 else 1

    @abstractmethod
    def apply(
        self, mask: Tensor, var: Tensor, flux: Flux, grid: Tensor, order: int
    ) -> Tensor:
        """Apply boundary conditions.

        Args:
            mask: mask of bc object
            var: Field vaules at the volume center
            flux: Flux object to apply the boundary condition
            grid: `Mesh.grid` to be used for `Callable` `self.bc_val`
            order: order of boundary evaluation. if `order` is zero, face value will bel linearly evaulated in between cell `i` and `i+1`. else `order` is non-zero, face value will be `order`-derivative at the face.
        """
        ...


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions."""

    def apply(
        self, mask: Tensor, var: Tensor, flux: Flux, grid: Tensor, order: int
    ) -> None:
        """Apply BC"""
        dim = var.size(0)

        if callable(self.bc_val):

            # TODO: NEED TO BE CHECKED
            bc_callable = self.bc_val(grid, mask)

            for d in range(dim):
                face_val = flux.face(d, self.bc_face)

                if order == 0:
                    face_val[mask] = bc_callable
                else:
                    # TODO: shit I need mesh info...
                    raise ValueError

                flux.to_face(d, self.bc_face[0], self.bc_face[1], face_val)

        else:
            for d in range(dim):

                face_val = flux.face(d, self.bc_face)

                if isinstance(self.bc_val, int) or isinstance(
                    self.bc_val, float
                ):
                    face_val[mask] = self.bc_val * self.face_n_dir
                else:
                    face_val[mask] = self.bc_val[d] * self.face_n_dir

                flux.to_face(d, self.bc_face[0], self.bc_face[1], face_val)


class Neumann(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(
        self, mask: Tensor, var: Tensor, flux: Flux, grid: Tensor
    ) -> None:
        """Apply BC
        Args:
            mask: mask of bc object
            var: Field vaules at the volume center
            flux: Flux object to apply the boundary condition
            grid: `Mesh.grid` to be used for `Callable` `self.bc_val`
        """
        dim = var.size(0)

        raise NotImplementedError


class Symmetry(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(
        self, mask: Tensor, var: Tensor, flux: Flux, grid: Tensor
    ) -> None:
        """Apply BC
        Args:
            mask: mask of bc object
            var: Field vaules at the volume center
            flux: Flux object to apply the boundary condition
            grid: `Mesh.grid` to be used for `Callable` `self.bc_val`
        """
        dim = var.size(0)
        for d in range(dim):
            if self.bc_face[1] == "l":
                opp_face = self.bc_face.replace("l", "r")
            else:
                opp_face = self.bc_face.replace("r", "l")

            face_val = flux.face(d, opp_face)
            flux.to_face(d, self.bc_face[0], self.bc_face[1], face_val)


class Periodic(BC):
    r"""Apply Periodic boundary conditions."""

    def apply(self, *_) -> None:
        """Apply BC"""
        # Do nothing
        pass


def _bc_val_type_check(bc_val: BC_val_type):

    if type(bc_val) not in get_args(BC_val_type):
        raise TypeError(
            f"BC: wrong bc variable -> {type(bc_val)} is not one of {get_args(BC_val_type)}!"
        )


BC_type = Union[Type[Dirichlet], Type[Neumann], Type[Symmetry], Type[Periodic]]


BC_FACTORY: dict[str, BC_type] = {
    "dirichlet": Dirichlet,
    "neumann": Neumann,
    "symmetry": Symmetry,
    "periodic": Periodic,
}
