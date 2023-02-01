#!/usr/bin/env python3
"""Module that contains boundary conditions for FDM method.

Supporting conditions:
    * Dirichlet
    * Neumann
    * Symmetry
    * Periodic

WIP:
    * Inflow/Outflow

Note:
    * The boundary conditions are applied in `["xl", "xr", "yl", "yr", "zl", "zr"]` order.

"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import get_args
from typing import Union

import torch
from torch import Tensor

from pyapes.core.backend import DType
from pyapes.core.geometry.basis import DIR_TO_NUM
from pyapes.core.geometry.basis import FDIR

BC_val_type = Union[
    int,
    float,
    list[int],
    list[float],
    Callable[[tuple[Tensor, ...], Tensor], Tensor],
    None,
]
"""BC value type."""
BC_config_type = dict[str, Union[str, BC_val_type]]
"""BC config type."""


@dataclass
class BC(ABC):
    """Abstract base class of the boundary condition object."""

    bc_id: str
    """BC id"""
    bc_val: BC_val_type
    """Boundary values."""
    bc_face: str
    """Face to assign bc. Convention follows `dim` + `l` or `r`. e.g. `x_l`"""
    bc_mask: Tensor
    """Mask of the mesh where to apply BCs."""
    bc_var_name: str
    """Target variable name."""
    dtype: DType
    """Data type. Since we do not store `Mesh` object here, explicitly pass dtype."""
    device: torch.device
    """Device. Since we do not store `Mesh` object here, explicitly pass device."""

    def __post_init__(self):

        _bc_val_type_check(self.bc_val)
        self._bc_face_dim = DIR_TO_NUM[self.bc_face[0]]

        if self.bc_face[-1] == "l":
            self._bc_n_dir: int = -1
        else:
            self._bc_n_dir: int = 1

        self._bc_type = self.__class__.__name__.lower()

        self._bc_mask_prev = torch.roll(
            self.bc_mask, -self.bc_n_dir, self.bc_face_dim
        )
        self._bc_mask_forward = torch.roll(
            self.bc_mask, self.bc_n_dir, self.bc_face_dim
        )

    @property
    def bc_mask_prev(self) -> Tensor:
        """Previous (one index before in `-self.bc_n_dir` direction) mask of the mesh where to apply BCs.

        Examples:
            >>> bc_mask = torch.Tensor([True, False, False, False])
            >>> bc_mask_prev = torch.Tensor([False, True, False, False])

        """
        return self._bc_mask_prev

    @property
    def bc_mask_forward(self) -> Tensor:
        """Forward (one index after in `self.bc_n_dir` direction) mask of the mesh where to apply BCs.

        Examples:
            >>> bc_mask = torch.Tensor([True, False, False, False])
            >>> bc_mask_prev = torch.Tensor([False, False, False, True])

        """
        return self._bc_mask_forward

    @property
    def bc_treat(self) -> bool:
        """Whether the special treatment is needed for discretization or rhs."""
        if self.bc_type == "neumann" or self.bc_type == "symmetry":
            return True
        else:
            return False

    @property
    def bc_type(self) -> str:
        """BC type."""
        return self._bc_type

    @property
    def bc_face_dim(self) -> int:
        """Dimension (e.g. `x`: `0`) that the bc is applied to. Used to roll the variable `Tensor`."""
        return self._bc_face_dim

    @property
    def bc_n_dir(self) -> int:
        """Normal direction of bc side. -1 for `l` side, 1 for `r` side."""
        return self._bc_n_dir

    @property
    def type(self) -> str:
        """BC type."""
        return self.__class__.__name__.lower()

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

    def laplacian_rhs(self, *_) -> None:
        # Do nothing since rhs is redefined anyway
        pass

    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int
    ) -> None:

        assert self.bc_val is not None, "BC: bc_val is not specified!"

        if callable(self.bc_val):
            var[var_dim, self.bc_mask] = self.bc_val(grid, self.bc_mask)
        elif isinstance(self.bc_val, list):
            var[var_dim, self.bc_mask] = self.bc_val[var_dim]
        else:
            var[var_dim, self.bc_mask] = self.bc_val


class Neumann(BC):
    r"""Apply Neumann boundary conditions.

    Note:
        - Gradient is calculated using the 1st order forward difference.
    """

    def laplacian_rhs(self, rhs: Tensor, grid):
        """Adjust RHS for Neumann BCs."""

        assert self.bc_val is not None, "BC: bc_val is not specified!"

        for i in range(rhs.shape[0]):
            mask = torch.roll(self.bc_mask, -self.bc_n_dir, i)
            if callable(self.bc_val):
                rhs[i][mask] += (
                    2
                    / 3
                    * self.bc_val(grid, self.bc_mask)
                    * float(-self.bc_n_dir)
                )
            elif isinstance(self.bc_val, list):
                rhs[i][mask] += 2 / 3 * self.bc_val[i] * float(-self.bc_n_dir)
            else:
                rhs[i][mask] += 2 / 3 * self.bc_val * float(-self.bc_n_dir)

    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int
    ) -> None:
        """Apply BC"""

        assert self.bc_val is not None, "BC: bc_val is not specified!"

        sign = float(self.bc_n_dir)

        dx = sign * (
            grid[self.bc_face_dim][self.bc_mask]
            - grid[self.bc_face_dim][self.bc_mask_prev]
        )

        if callable(self.bc_val):
            c_bc_val = self.bc_val(grid, self.bc_mask)
            var[var_dim, self.bc_mask] = (
                dx * c_bc_val + var[var_dim, self.bc_mask_prev]
            )
        elif isinstance(self.bc_val, list):
            var[var_dim, self.bc_mask] = (
                dx * self.bc_val[var_dim] + var[var_dim, self.bc_mask_prev]
            )
        else:
            var[var_dim, self.bc_mask] = (
                dx * self.bc_val + var[var_dim, self.bc_mask_prev]
            )


class Symmetry(BC):
    r"""Apply Neumann boundary conditions."""

    def laplacian_rhs(self, *_) -> None:
        # Do nothing since gradient is zero
        pass

    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int
    ) -> None:

        assert grid

        var[var_dim, self.bc_mask] = var[var_dim, self.bc_mask_prev]


class Periodic(BC):
    r"""Apply Periodic boundary conditions."""

    def laplacian_rhs(self, *_) -> None:
        raise NotImplementedError

    def apply(
        self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int
    ) -> None:

        assert grid

        var[var_dim, self.bc_mask] = var[var_dim, self.bc_mask_forward]


def _bc_val_type_check(bc_val: BC_val_type):
    """Check whether the bc_val is of the correct type."""

    if not isinstance(bc_val, Callable):

        if type(bc_val) not in get_args(BC_val_type):
            raise TypeError(
                f"BC: wrong bc variable -> {type(bc_val)} is not one of {get_args(BC_val_type)}!"
            )


def mixed_bcs(
    bc_val: list[float | Callable[[tuple[Tensor, ...], Tensor], Tensor]],
    bc_type: list[str],
):
    """Simple pre-defined boundary condition.

    Warning:
        - Only works for `Box` type object.

    Args:
        dim (int): dimension of mesh
        bc_val (list[float|Callable]): values at boundaries.
        bc_type (list[str]): Types of bcs

    Returns:
        list[BC_config_type]: list of dictionary used to declare the boundary conditions.
    """

    bc_config = []
    for i, (v, t) in enumerate(zip(bc_val, bc_type)):
        bc_config.append({"bc_face": FDIR[i], "bc_type": t, "bc_val": v})
    return bc_config


def homogeneous_bcs(
    dim: int, bc_val: float | list[float] | None, bc_type: str
) -> list[BC_config_type]:
    """Simple pre-defined boundary condition.

    Warning:
        - Only works for `Box` type object.

    Args:
        dim: dimension of mesh
        bc_val: value at the boundary. Homogenous at the boundary surface.
        bc_type: Type of bc

    Returns:
        list[BC_config_type]: list of dictionary used to declare the boundary conditions.
    """

    bc_config = []
    for i in range(dim * 2):
        bc_config.append(
            {
                "bc_face": FDIR[i],
                "bc_type": bc_type,
                "bc_val": bc_val[i] if isinstance(bc_val, list) else bc_val,
            }
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


BC_FACTORY: dict[
    str, type[Dirichlet] | type[Neumann] | type[Symmetry] | type[Periodic]
] = {
    "dirichlet": Dirichlet,
    "neumann": Neumann,
    "symmetry": Symmetry,
    "periodic": Periodic,
}
