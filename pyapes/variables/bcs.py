#!/usr/bin/env python3
"""Module that contains boundary conditions for FDM method.

Supporting conditions:
    * Dirichlet
    * Neumann
    * Symmetry
    * Periodic

Note:
    * The boundary face is identified by `face_normal_dim` + `side` e.g. `["xl", "xu", "yl", "yu", "zl", "zu"]`.
    * Or ["rl", "ru", "zl", "zu"] for rz coordinate.

"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import get_args
from typing import NamedTuple
from typing import TypedDict

import torch
from torch import Tensor

from pyapes.backend import DType
from pyapes.geometry.basis import DIR_TO_NUM
from pyapes.geometry.basis import DIR_TO_NUM_RZ
from pyapes.geometry.basis import FDIR
from pyapes.geometry.basis import FDIR_RZ

BC_val_type = int | float | list[int] | list[float] | Callable | Tensor | None
"""BC value type."""


class BCConfig(TypedDict):
    """BC config type."""

    bc_face: str
    bc_type: str
    bc_val: BC_val_type
    bc_val_opt: dict[str, Tensor] | None


@dataclass
class BC(ABC):
    """Abstract base class of the boundary condition object."""

    bc_id: str
    """BC id"""
    bc_val: BC_val_type
    """Boundary values."""
    bc_val_opt: dict[str, Tensor] | None
    """Boundary value options. Can be additional `Tensor` during the computation of BCs."""
    bc_face: str
    """Face to assign bc. Convention follows `dim` + `l` or `r`. e.g. `x_l`"""
    bc_mask: Tensor
    """Mask of the mesh where to apply BCs."""
    bc_var_name: str
    """Target variable name."""
    bc_coord_sys: str
    """Boundary coordinate system. Currently, either `xyz` or `rz`."""
    mesh_dim: int
    """Mesh dimension."""
    dtype: DType
    """Data type. Since we do not store `Mesh` object here, explicitly pass dtype."""
    device: torch.device
    """Device. Since we do not store `Mesh` object here, explicitly pass device."""

    def __post_init__(self):
        _bc_val_type_check(self.bc_val)
        if self.bc_coord_sys == "rz":
            self._bc_face_dim = DIR_TO_NUM_RZ[self.bc_face[0]]
        else:
            self._bc_face_dim = DIR_TO_NUM[self.bc_face[0]]

        if self.bc_face[-1] == "l":
            self._bc_n_dir: int = -1
        else:
            self._bc_n_dir: int = 1

        self._bc_type = self.__class__.__name__.lower()

        self._bc_mask_prev = torch.roll(self.bc_mask, -self.bc_n_dir, self.bc_face_dim)
        self._bc_mask_prev2 = torch.roll(
            self.bc_mask, -self.bc_n_dir * 2, self.bc_face_dim
        )
        self._bc_mask_forward = torch.roll(
            self.bc_mask, self.bc_n_dir, self.bc_face_dim
        )
        self._bc_mask_forward2 = torch.roll(
            self.bc_mask, self.bc_n_dir * 2, self.bc_face_dim
        )
        self._bc_n_vec = torch.zeros(3, dtype=self.dtype.float, device=self.device)
        self._bc_n_vec[self.bc_face_dim] = self.bc_n_dir

    def bc_mask_shift(self, shift: int) -> Tensor:
        """Shift `bc_mask` by `shift` indices in the direction of `self.bc_face_dim`.
        Negative `shift` will return previous location, while positive `shift` will return forward location.

        Example:
            >>> self.bc_mask = torch.tensor([True, False, False, False])
            >>> self.bc_mask_shift(-1)
            torch.tensor([False, True, False, False])
        """
        return torch.roll(self.bc_mask, shift, self.bc_face_dim)

    @property
    def bc_n_vec(self) -> Tensor:
        """Return normal vector of the bc surface."""
        return self._bc_n_vec

    @property
    def bc_mask_prev2(self) -> Tensor:
        """Previous (two indices before in `-self.bc_n_dir` direction) mask of the mesh where to apply BCs.

        Examples:
            >>> bc_mask = torch.Tensor([True, False, False, False])
            >>> bc_mask_prev = torch.Tensor([False, False, True, False])

        """
        return self._bc_mask_prev2

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
            >>> bc_mask_forward = torch.Tensor([False, False, False, True])

        """
        return self._bc_mask_forward

    @property
    def bc_mask_forward2(self) -> Tensor:
        """Forward (one index after in `self.bc_n_dir` direction) mask of the mesh where to apply BCs.

        Examples:
            >>> bc_mask = torch.Tensor([True, False, False, False])
            >>> bc_mask_forward2 = torch.Tensor([False, False, True, False])

        """
        return self._bc_mask_forward2

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
    def apply(self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int) -> None:
        """Apply BCs.

        Args:
            var: variable to apply BCs
            grid: grid of the mesh
            var_dim: dimension of the variable
        """
        ...


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions."""

    def apply(self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int) -> None:
        assert self.bc_val is not None, "BC: bc_val is not specified!"

        if callable(self.bc_val):
            at_bc = self.bc_val(grid, self.bc_mask, var, self.bc_val_opt)
            var[var_dim, self.bc_mask] = at_bc
        elif isinstance(self.bc_val, list):
            var[var_dim, self.bc_mask] = self.bc_val[var_dim]
        elif isinstance(self.bc_val, int | float):
            var[var_dim, self.bc_mask] = float(self.bc_val)
        elif isinstance(self.bc_val, Tensor):
            var[var_dim, self.bc_mask] = self.bc_val
        else:
            raise TypeError("Dirichlet: bc_val must be float, int, callable or list!")


class Neumann(BC):
    r"""Apply Neumann boundary conditions.

    Note:
        - Gradient is calculated using the 1st order forward difference.
    """

    def apply(self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int) -> None:
        """Apply Neumann BC in the second-order accuracy."""

        assert self.bc_val is not None, "BC: bc_val is not specified!"

        dx = (
            grid[self.bc_face_dim][self.bc_mask]
            - grid[self.bc_face_dim][self.bc_mask_prev]
        )

        var_p = var[var_dim][self.bc_mask_prev]
        var_pp = var[var_dim][self.bc_mask_prev2]

        # Neumann BC:
        # Second order forward-backward difference gives
        # p0 = 4/3 p1 - 1/3 p2 - 2/3 V dx
        # pN = 4/3 pN-1 - 1/3 pN-2 + 2/3 V dx
        if callable(self.bc_val):
            c_bc_val = self.bc_val(grid, self.bc_mask, var, self.bc_val_opt)
        elif isinstance(self.bc_val, list):
            c_bc_val = self.bc_val[var_dim]
        elif isinstance(self.bc_val, float | int):
            c_bc_val = float(self.bc_val)
        elif isinstance(self.bc_val, Tensor):
            c_bc_val = self.bc_val
        else:
            raise TypeError("Neumann: bc_val must be float, int, callable or list!")

        var[var_dim][self.bc_mask] = (
            4 / 3 * var_p - 1 / 3 * var_pp + 2 / 3 * c_bc_val * dx * self.bc_n_dir
        )


class Symmetry(BC):
    r"""Apply Neumann boundary conditions."""

    def apply(self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int) -> None:
        assert grid

        var[var_dim, self.bc_mask] = var[var_dim, self.bc_mask_prev]


class Periodic(BC):
    r"""Apply Periodic boundary conditions."""

    def apply(self, var: Tensor, grid: tuple[Tensor, ...], var_dim: int) -> None:
        assert grid

        if self.bc_n_dir < 0:
            # Left hand side take other side's value
            var_p = var[var_dim, self.bc_mask_prev]
            var_f = var[var_dim, self.bc_mask_forward]
            var_ff = var[var_dim, self.bc_mask_forward2]
            var[var_dim, self.bc_mask] = var_p - var_f + var_ff

        else:
            # Right hand side keep its value
            var[var_dim, self.bc_mask] = var[var_dim, self.bc_mask_forward]


def _bc_val_type_check(bc_val: BC_val_type):
    """Check whether the bc_val is of the correct type."""

    if not isinstance(bc_val, Callable):
        if type(bc_val) not in get_args(BC_val_type):
            raise TypeError(
                f"BC: wrong bc variable -> {type(bc_val)} is not one of {get_args(BC_val_type)}!"
            )


class BCContainer(TypedDict, total=False):
    """Type of dictionary used to declare the boundary conditions."""

    bc_type: str
    bc_val: BC_val_type
    bc_val_opt: dict[str, Tensor] | None


class CylinderBoundary(NamedTuple):
    """Setup interface for the configuration of the cylinder type boundary conditions.

    Example:
        >>> f_bc = BoxBoundary(
            rl={"bc_type": "symmetric", "bc_val": None},
            rr={"bc_type": "neumann", "bc_val": 0},
            zl={"bc_type": "periodic", "bc_val": None},
            zr={"bc_type": "dirichlet", "bc_val": 0.44},
        )
        >>> f_bc()
        # Will return list[BC_config_type] (None entry will be ignored)
        [
            {"bc_face": "rl", "bc_type": "symmetric", "bc_val": None},
            {"bc_face": "rr", "bc_type": "neumann", "bc_val": 0},
            {"bc_face": "zl", "bc_type": "periodic", "bc_val": None},
            {"bc_face": "zr", "bc_type": "dirichlet", "bc_val": 0.44},
        ]

    """

    rl: BCContainer | None = None
    ru: BCContainer | None = None
    zl: BCContainer | None = None
    zu: BCContainer | None = None

    def __call__(self) -> list[BCConfig]:
        return _get_bc_dict(self, FDIR_RZ)


class BoxBoundary(NamedTuple):
    """Setup interface for the configuration of the box type boundary conditions.

    Example:
        >>> f_bc = BoxBoundary(
            xl={"bc_type": "dirichlet", "bc_val": 0.44},
            xr={"bc_type": "neumann", "bc_val": 0},
            yl={"bc_type": "periodic", "bc_val": None},
            yr={"bc_type": "symmetric", "bc_val": None},
        )
        >>> f_bc()
        # Will return list[BC_config_type] (None entry will be ignored)
        [
            {"bc_face": "xl", "bc_type": "dirichlet", "bc_val": 0.44},
            {"bc_face": "xr", "bc_type": "neumann", "bc_val": 0},
            {"bc_face": "yl", "bc_type": "periodic", "bc_val": None},
            {"bc_face": "yr", "bc_type": "symmetric", "bc_val": None},
        ]

    """

    xl: BCContainer | None = None
    xu: BCContainer | None = None
    yl: BCContainer | None = None
    yu: BCContainer | None = None
    zl: BCContainer | None = None
    zu: BCContainer | None = None

    def __call__(self) -> list[BCConfig]:
        return _get_bc_dict(self, FDIR)


def _get_bc_dict(
    bc_config: CylinderBoundary | BoxBoundary, fdir: list[str]
) -> list[BCConfig]:
    config: list[BCConfig] = []

    for face in fdir:
        bc_dict = getattr(bc_config, face)
        if bc_dict is not None:
            config.append(
                {
                    "bc_face": face,
                    "bc_type": bc_dict["bc_type"],
                    "bc_val": bc_dict["bc_val"],
                    "bc_val_opt": bc_dict["bc_val_opt"]
                    if "bc_val_opt" in bc_dict
                    else None,
                }
            )

    return config


def mixed_bcs(
    bc_val: list[BC_val_type],
    bc_type: list[str],
) -> list[BCConfig]:
    """Simple pre-defined boundary condition.

    Warning:
        - Only works for `Box` type object with out any `bc_val_opt`.

    Args:
        dim (int): dimension of mesh
        bc_val (BC_val_type): values at boundaries.
        bc_type (list[str]): Types of bcs

    Returns:
        list[BC_config_type]: list of dictionary used to declare the boundary conditions.
    """

    bc_config: list[BCConfig] = []
    for i, (v, t) in enumerate(zip(bc_val, bc_type)):
        bc_config.append(
            {"bc_face": FDIR[i], "bc_type": t, "bc_val": v, "bc_val_opt": None}
        )
    return bc_config


def homogeneous_bcs(
    dim: int,
    bc_val: float | list[float] | list[list[float]] | None,
    bc_type: str,
) -> list[BCConfig]:
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
                "bc_val_opt": None,
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


BC_type = Dirichlet | Neumann | Symmetry | Periodic


BC_FACTORY: dict[
    str, type[Dirichlet] | type[Neumann] | type[Symmetry] | type[Periodic]
] = {
    "dirichlet": Dirichlet,
    "neumann": Neumann,
    "symmetry": Symmetry,
    "periodic": Periodic,
}
