#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM) """
from typing import Any
from typing import Union

import torch
from torch import Tensor

from pyABC.core.variables import Field

DIR = ["x", "y", "z"]
FDIR = ["xl", "xr", "yl", "yr", "zl", "zr"]

DIR_TO_NUM: dict[str, int] = {"x": 0, "y": 1, "z": 2}


class Flux:
    """Flux container.

    Note:
        - To distinguis leading index and other index, leading is int and other is str.

    >>> flux_tensor = torch.tensor(...)
    >>> flux = Flux()
    >>> flux.add(i, j, flux_tensor)  # j -> DIR[j] -> "x" or "y" or "z", i -> 0 or 1 or 2.
    >>> flux(0, "x")  # return flux_tensor
    """

    def __init__(self):

        self._center: dict[int, dict[str, Tensor]] = {}
        self._face: dict[int, dict[str, dict[str, Tensor]]] = {}

    def to_center(self, i: int, j: str, T: Tensor):
        """Add centervalued as dictionary."""

        try:
            self._center[i][j] = T
        except KeyError:
            self._center.update({i: {j: T}})

    def __call__(self, i: int, j: str) -> Tensor:
        """Return flux values with parentheses."""

        if j in DIR:
            return self._center[i][j]
        else:
            assert j in FDIR, f"Flux: face index should be one of {FDIR}"
            return self._face[i][j[0]][j[1]]

    @property
    def c_idx(self) -> tuple[list[int], list[str]]:
        """Return center index."""

        idx_i = list(self._center.keys())
        idx_j = list(self._center[0].keys())

        return (idx_i, idx_j)

    def face(self, i: int, f_idx: str) -> Tensor:
        """Return face value with index."""

        assert f_idx in FDIR, f"Flux: face index should be one of {FDIR}!"

        return self._face[i][f_idx[0]][f_idx[1]]

    def to_face(self, i: int, j: str, f: str, T: Tensor) -> None:
        """Assign face values to `self._face`.

        Args:
            i (int): leading index
            j (str): dummy index (to be summed)
            f (str): face index l (also for back and bottom), r (also for front and top)
            T (Tensor): face values to be stored.
        """

        if i in self._face:
            if j in self._face[i]:
                self._face[i][j][f] = T
            else:
                self._face[i].update({j: {f: T}})
        else:
            self._face.update({i: {j: {f: T}}})

    def flux_sum(self) -> None:

        self._center = {}

        for i in self._face:
            c_val = {}
            for j in self._face[i]:
                c_val.update(
                    {j: (self._face[i][j]["l"] + self._face[i][j]["r"]) / 2}
                )
            self._center[i] = c_val

    def __mul__(self, target: Union[float, int, Field]) -> Any:
        """Multiply coeffcient to the flux"""

        if isinstance(target, float) or isinstance(target, int):
            for i in self._face:
                for j in self._face[i]:
                    self._face[i][j]["l"] *= target
                    self._face[i][j]["r"] *= target
                    try:
                        self._center[i][j] *= target
                    except KeyError:
                        self.flux_sum()
                        self._center[i][j] *= target

        elif isinstance(target, Field):
            # Multiply tensor in j direction for self._center
            # Will be used for u_j \nabla_j u_i
            for i in self._face:
                for j in self._face[i]:
                    self._center[i][j] *= target()
        else:
            raise TypeError("Flux: wrong input type is given!")

        return self


class Discretizer:
    """Base class of FVM discretization.

    Examples:

        >>> # Laplacian of scalar field same for Div
        >>> laplacian = Laplacian()
        >>> res = laplacian(coeff, phi)
        >>> res.flux(0, "xl")    # d^2 phi/dx_1^2 on the left side of x directional cell face
        >>> res.flux_sum()
        >>> res.flux(0, "x")     # averaged cell centered value in x
    """

    # Class member
    ops: dict[int, dict[str, Union[Flux, str]]] = {}
    rhs: Union[Tensor, float]

    @property
    def var(self) -> Field:
        raise NotImplementedError

    @property
    def flux(self) -> Flux:
        raise NotImplementedError

    def set_config(self, config: dict):

        self.config = config

    def __add__(self, other: Any) -> Any:

        # order of multiple add?
        self.ops.update(
            {0: {"flux": self.flux, "op": self.__class__.__name__}}
        )

        idx = list(self.ops.keys())
        self.ops.update(
            {idx[-1] + 1: {"flux": other.flux, "op": other.__class__.__name__}}
        )

        return self

    def __sub__(self, other: Any) -> Any:

        if len(self.ops) == 0:
            self.ops.update(
                {0: {"flux": self.flux * -1, "op": self.__class__.__name__}}
            )

        idx = list(self.ops.keys())
        self.ops.update(
            {
                idx[-1]
                + 1: {"flux": other.flux * -1, "op": other.__class__.__name__}
            }
        )

        return self

    def __eq__(self, other: Union[Tensor, float]) -> Any:

        if isinstance(other, Tensor):
            self.rhs = other
        else:
            self.rhs = torch.zeros_like(self.var()) + other

        return self


class Ddt(Discretizer):
    """Time discretization."""

    def __call__(self, var: Field) -> Any:

        raise NotImplementedError


class Grad(Discretizer):
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Args:
        var: Field object to be discretized ($\Phi$).

    """

    def __call__(self, var: Field) -> Flux:

        self._var = var

        dx = var.mesh.dx

        grad = Flux()

        for i in range(var.dim):

            for j in range(var.dim):

                grad.to_center(
                    i,
                    DIR[j],
                    (torch.roll(var()[i], -1, j) - torch.roll(var()[i], 1, j))
                    / (2 * dx[j]),
                )

        self._flux = grad

        return grad

    @property
    def var(self) -> Field:
        return self._var

    @property
    def flux(self) -> Flux:
        return self._flux


class Div(Discretizer):
    r"""Divergence

        \frac{\partial}{\partial x_j}
        \left(
            u_j \phi_i
        \right)

    Args:
        var_i: Field object to be discretized ($\Phi_i$)
        var_j: convective variable ($\vec{u}_j$)
    """

    def __call__(self, var_i: Field, var_j: Field) -> Any:

        self._var = var_i

        flux = Flux()

        for i in range(var_i.dim):
            for j in range(var_j.dim):

                vi_c = var_i()
                vj_c = var_j()

                fl = (torch.roll(vj_c, 1, j) * vi_c + vj_c * vi_c) / (
                    2 * var_j.dx[j]
                )
                flux.to_face(i, DIR[j], "l", fl)

                fr = (torch.roll(vj_c, -1, j) * vi_c + vj_c * vi_c) / (
                    2 * var_j.dx[j]
                )
                flux.to_face(i, DIR[j], "r", fr)

        # Apply bc later on...
        self._flux = flux
        return self

    @property
    def var(self) -> Field:
        return self._var

    @property
    def flux(self) -> Flux:
        return self._flux


class Laplacian(Discretizer):
    r"""Variable discretization: Laplacian

    .. math::

        \frac{\partial}{\partial x_j}
        \left(
            \Gamma^\Phi \frac{\partial \Phi}{\partial x_j}
        \right)


    Args:
        coeff: coefficient of the Laplacian operator ($\Gamma^\Phi$)
        var: Field object to be discretized ($\Phi$)
    """

    def __call__(self, coeff: float, var: Field) -> Any:

        self._var = var

        fvm_Grad = Grad()
        grad = fvm_Grad(var)
        flux = Flux()

        for i in range(var.dim):
            for j in range(var.dim):

                fl = (
                    (torch.roll(grad(0, DIR[j]), 1, j) + grad(0, DIR[j]))
                    / (2 * var.dx[j])
                    * coeff
                )
                flux.to_face(i, DIR[j], "l", fl)
                fr = (
                    (torch.roll(grad(0, DIR[j]), -1, j) + grad(0, DIR[j]))
                    / (2 * var.dx[j])
                    * coeff
                )
                flux.to_face(i, DIR[j], "r", fr)

        self._flux = flux

        return self

    @property
    def var(self) -> Field:
        return self._var

    @property
    def flux(self) -> Flux:
        return self._flux


def _flux_linear(flux: Flux, f_var: Tensor) -> Flux:
    r"""Linear interpolation of the flux from the node values.

    .. math:

        \Phi_f = \frac{\Phi^{+1} + \Phi^{c}}{2}

    Args:
        flux: flux object to be calculated
        f_var: field values at the cell center
    """

    raise NotImplementedError


def _fvm_i_bc_apply(var: Field, flux: Flux, type: str = "grad") -> Flux:
    """Apply BC for the product of a single variable.

    Args:
        var: target variable. Same Variable type with the final return type
        flux: flux calculated from nodes

    Returns:
        Boundary assigned Flux
    """

    raise NotImplementedError
