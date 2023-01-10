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
from pyapes.core.mesh import Mesh

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
    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int
    ) -> None:
        """Apply BCs.

        Args:
            var: variable to apply BCs
            grid: grid of the mesh
            var_dim: dimension of the variable
        """
        ...


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions."""

    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int
    ) -> None:

        assert not isinstance(
            self.bc_val, dict
        ), f"BC: {self.__class__.__name__} does not support dict type boundary value."

        dim = var.size(0)

        if callable(self.bc_val):
            var[var_dim, self.bc_mask] = self.bc_val(
                grid, self.bc_mask, var_dim
            )
        elif isinstance(self.bc_val, list):
            var[var_dim, self.bc_mask] = self.bc_val[var_dim]
        else:
            var[var_dim, self.bc_mask] = self.bc_val


class Neumann(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim=int
    ) -> None:
        """Apply BC"""

        raise NotImplementedError


class Symmetry(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(self, var: Tensor, *_) -> None:
        """Apply BC"""
        dim = var.size(0)

        for d in range(dim):

            if self.bc_face[1] == "l":
                opp_face = self.bc_face.replace("l", "r")
            else:
                opp_face = self.bc_face.replace("r", "l")


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


# WIP
def obstacle_mask(mesh: Mesh, masks: dict) -> tuple[dict, dict]:
    """Create a mask from the objects (self.mesh.objs).

    Warning:
        - Currently only works for the Box type obstacle.
    """
    x = mesh.x
    dx = mesh.dx
    nx = mesh.nx
    dim = mesh.dim

    obstacle = mesh.obstacle

    dtype = mesh.dtype
    device = mesh.device

    mask_to_save = dict()
    mask_set = dict()

    if obstacle is not None:

        mask = torch.zeros(*nx, dtype=dtype.bool, device=device)

        for i, obj in enumerate(obstacle):

            mask_obj = torch.zeros(*nx, dtype=dtype.bool, device=device)
            get_box_mask(x, dx, obj.lower, obj.upper, dtype, mask_obj, dim)

            mask += mask_obj
            mask_set[f"ob-{i}"] = mask_obj

        for m in masks:

            if "Obstacle" in m:
                obj_wall_m = masks[m]
                mask[obj_wall_m] = False

        # Save as sub dictionary
        # Should id be face dir?
        mask_to_save["Obstacle-all"] = mask

    else:
        mask_to_save["Obstacle-all"] = mask = torch.zeros(
            *nx, dtype=dtype.bool, device=device
        )

    return mask_set, mask_to_save


def domain_mask(
    mesh: Mesh, var_dim: int
) -> tuple[dict[int, dict[str, Tensor]], dict[int, dict[str, Tensor]]]:
    """Create a mask from the objects (self.mesh.objs).

    Warning:
        - Currently only works for the `Box` object.

    Args:
        mesh (Mesh): Mesh object
        var_dim (int): Variable dimension

    Returns:
        tuple[dict[int, dict[str, Tensor]], dict[int, dict[str, Tensor]]]: mask_to_save. Here, `int` is the variable dimension and `str` is the geometry component id and `Tensor` is the mask.
    """

    x = mesh.x
    dx = mesh.dx
    nx = mesh.nx
    dim = mesh.dim

    domain = mesh.domain.config
    obstacle = mesh.obstacle

    dtype = mesh.dtype
    device = mesh.device

    mask_to_save: dict[int, dict[str, Tensor]] = {}
    obj_mask_sep: dict[int, dict[str, Tensor]] = {}

    # Loop over patch objects
    for d in range(var_dim):
        for obj in domain:

            mask = torch.zeros(*nx, dtype=dtype.bool, device=device)
            mask = get_patch_mask(x, dx, domain[obj], mask, dim)

            if len(mask_to_save) == 0:
                mask_to_save[d] = {f"d-{domain[obj]['face']}": mask}
            else:
                mask_to_save[d].update({f"d-{domain[obj]['face']}": mask})

        if obstacle is not None:

            for idx, geo in enumerate(obstacle):

                for obj in geo.config:

                    mask = torch.zeros(*nx, dtype=dtype.bool, device=device)
                    mask = get_patch_mask(x, dx, geo.config[obj], mask, dim)

                    mask_to_save[d].update(
                        {f"o{idx}-{geo.config[obj]['face']}": mask}
                    )
                    if len(obj_mask_sep) == 0:
                        obj_mask_sep[d] = {
                            f"o{idx}-{geo.config[obj]['face']}": mask
                        }
                    else:
                        obj_mask_sep[d].update(
                            {f"o{idx}-{geo.config[obj]['face']}": mask}
                        )

    return mask_to_save, obj_mask_sep


def get_box_mask(
    x: list[Tensor],
    dx: Tensor,
    lower: list[float],
    upper: list[float],
    dtype: DType,
    mask: Tensor,
    dim: int,
) -> Tensor:
    """Get inner box object mask.

    Args:
        x: grid coordinates.
        dx: grid spacing
        lower: lower bound of the box object
        upper: upper bound of the box object
        dtype: data type to be return
        mask: mask to be returned
        dim: dimension of the mesh
    """

    _nx = torch.zeros(dim, dtype=dtype.int)
    _ix = torch.zeros_like(_nx)

    for i in range(dim):
        x_p = x[i][torch.argmin(abs(x[i] - lower[i]))]
        edge = torch.tensor(upper[i] - lower[i], dtype=dtype.float)
        _nx[i] = torch.ceil(edge / dx[i]).type(torch.long) + 1
        _ix[i] = torch.argmin(abs(x[i] - x_p))

    mask = _assign_mask(mask, _ix, _nx, dim)

    return mask


def get_patch_mask(
    x: list[Tensor], dx: Tensor, obj: dict, mask: Tensor, dim: int
) -> Tensor:
    """Get mask for each surface (side) of geometry.

    Args:
        x (list[Tensor]): grid coordinates.
        dx (Tensor): grid spacing
        obj (dict): surface configuration contains `x_p` and `e_x`
        mask (Tensor): mask to be returned
        dim (int): dimension of the mesh

    Returns:
        Tensor: mask
    """

    _nx = torch.zeros(dim, dtype=torch.long)
    _ix = torch.zeros_like(_nx)

    x_p = obj["x_p"]
    e_x = obj["e_x"]

    for i in range(dim):
        x_p[i] = x[i][torch.argmin(abs(x[i] - x_p[i]))]
        _nx[i] = torch.ceil(e_x[i] / dx[i]).type(torch.long) + 1
        _ix[i] = torch.argmin(abs(x[i] - x_p[i]))

    mask = _assign_mask(mask, _ix, _nx, dim)

    return mask


def _assign_mask(mask: Tensor, ix: Tensor, nx: Tensor, dim: int) -> Tensor:
    """Assign boolean value to the mask `Tensor`.

    Args:
        mask (Tensor): mask to be returned
        ix (Tensor): index to assign the mask value
        nx (Tensor): offset to assign the mask value
        dim (int): dimension of the mesh

    Returns:
        Tensor: mask
    """

    if dim == 1:
        mask[
            ix[0] : ix[0] + nx[0],
        ] = True
    elif dim == 2:
        mask[
            ix[0] : ix[0] + nx[0],
            ix[1] : ix[1] + nx[1],
        ] = True
    else:
        mask[
            ix[0] : ix[0] + nx[0],
            ix[1] : ix[1] + nx[1],
            ix[2] : ix[2] + nx[2],
        ] = True

    return mask
