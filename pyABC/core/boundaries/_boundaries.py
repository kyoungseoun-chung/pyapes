#!/usr/bin/env python3
"""Field boundary conditions.

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
from typing import Union

import torch
from torch import Tensor

from pyABC.core.backend import DType
from pyABC.core.geometry.basis import DimOrder
from pyABC.core.geometry.basis import NormalDir
from pyABC.core.mesh import Mesh


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

    object_type: str
    bc_id: str
    bc_obj: str
    bc_val: Union[int, float, Callable]
    bc_var_name: str
    bc_face: str
    dtype: DType
    device: torch.device

    def __post_init__(self):

        # Check BC val type
        assert (
            type(self.bc_val) == float
            or type(self.bc_val) == int
            or callable(self.bc_val)
            or self.bc_val is None
        )

        assert self.object_type in ["domain", "obstacle"]

        # Sign of the normal vector
        self.bc_n_dir = NormalDir(self.bc_face)
        # Dimension of the bc_val to be applied
        self.bc_v_dim = DimOrder(self.bc_face)

        # Normal vector
        self.bc_n_vec = torch.zeros(
            3, device=self.device.type, dtype=self.dtype.float
        )
        self.bc_n_vec[self.bc_v_dim] = self.bc_n_dir

    @abstractmethod
    def apply(self) -> Tensor:
        """Apply boundary conditions."""


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions."""

    def apply(self, mask: Tensor, var: Tensor, mesh: Mesh) -> Tensor:
        """Dirichlet BC for finite different method.
        We do not use flux, rather directly use Variables.VAR and
        assign self.bc_val using the mask defined.

        Note:
            - This is only used for the scalar field since it appears in
              the Poisson equation.

        Args:
            mask: mask of bc object
            var: Variable where BC is applied
        """

        if callable(self.bc_val):
            var[0, mask] = self.bc_val(mesh, mask)
        else:
            var[0, mask] = self.bc_val

        return var


class Neumann(BC):
    r"""Apply Neumann boundary conditions.
    Use the first-order finite difference to calcualte the gradient at
    the boundary.
    """

    def apply(self, mask: Tensor, var: Tensor, mesh: Mesh) -> Tensor:

        mask_prev = torch.roll(mask, -self.bc_n_dir, self.bc_v_dim)

        dx = mesh.dx[self.bc_v_dim]

        if callable(self.bc_val):
            c_bc_val = self.bc_val(mesh, mask)
            var[0, mask] = dx * c_bc_val + var[0, mask_prev]
        else:
            var[0, mask] = dx * self.bc_val + var[0, mask_prev]

        return var


class Symmetry(BC):
    """Symmetry boundary condition."""

    def apply(self, mask: Tensor, var: Tensor, *_) -> Tensor:

        # Mast at the previous
        mask_prev = torch.roll(mask, self.bc_n_dir, self.bc_v_dim)

        var[0, mask] = var[0, mask_prev]

        return var


class Periodic(BC):
    """Periodic boundary condition."""

    def apply(self, mask: Tensor, var: Tensor, *_) -> Tensor:

        # Mast at the opposite location
        mask_prev = torch.roll(mask, -self.bc_n_dir, self.bc_v_dim)

        var[0, mask] = var[0, mask_prev]

        return var


BC_TYPE_FACTORY = {
    "dirichlet": Dirichlet,
    "neumann": Neumann,
    "symmetry": Symmetry,
    "periodic": Periodic,
}
