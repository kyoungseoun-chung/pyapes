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

    _center: dict[int, dict[str, Tensor]] = {}
    _face: dict[int, dict[str, dict[str, Tensor]]] = {}

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

        idx_i = list(self._center.keys())
        idx_j = list(self._center[0].keys())

        return (idx_i, idx_j)

    def face(self, i: int, f_idx: str) -> Tensor:

        assert f_idx in FDIR, f"Flux: face index should be one of {FDIR}!"

        return self._face[i][f_idx[0]][f_idx[1]]

    def to_face(self, i: int, j: str, f: str, T: Tensor) -> None:

        if i in self._face:
            if j in self._face[i]:
                self._face[i][j][f] = T
            else:
                self._face[i].update({j: {f: T}})
        else:
            self._face.update({i: {j: {f: T}}})

    def __mul__(self, coeff: float) -> Any:
        """Multiply coeffcient to the flux"""

        for i in self._face:
            for j in self._face[i]:

                self._face[i][j]["l"] *= coeff
                self._face[i][j]["r"] *= coeff

        return self


class Discretizer:
    """Base class of FVM discretization.

    Args:
        flux: flux object as a result of FVM operation. `len(flux)` corresponding to
              the dimension of the variable to be discretized.
        flux_total: will be used for the FVM.set_eq (not yet fixed)
    """

    # Class member
    flux: Flux
    ops: dict[int, dict[str, Union[Flux, str]]] = {}

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

        idx = list(self.ops.keys())

        if len(idx) == 0:
            self.ops.update({0: -1 * other.flux})
        else:
            self.ops.update({idx[-1] + 1: -1 * other.flux})

        return self


class Ddt(Discretizer):
    """Time discretization."""

    def __call__(self, var: Field) -> Any:

        raise NotImplementedError


class Grad(Discretizer):
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Examples:

        >>> # Gradient of scalar field
        >>> res = fvm_grad(phi)
        >>> res(0, "x")    # dphi/dx_1
        >>> # Gradient of vector field
        >>> res = fvm_grad(u)
        >>> res(0, "x")    # du_1/dx_1

    Args:
        var: Variable object to be discretized ($\Phi$).
            - If var.bcs is None, Grad operation is done by finite difference method.

    Returns:
        dictionary contains gradient of `var`.
    """

    def __call__(self, var: Field) -> Flux:

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

        return grad


class Div(Discretizer):
    r"""Divergence

        \frac{\partial}{\partial x_j}
        \left(
            u_j \phi_i
        \right)

    Examples:

        >>> # Advection of scalar field
        >>> res = fvm_div(phi, u)
        >>> res[0].x    # dphi/dx_1
        >>> res[0].c    # dphi/dx_1 + dphi/dx_2 + dphi/dx_3
        >>> # Advection of vector field
        >>> res = fvm_div(u, u)
        >>> res[0].x    # du_1/dx_1
        >>> res[0].c    # du_1/dx_1 + du_1/dx_2 + du_1/dx_3

    Args:
        var_i: Variable object to be discretized ($\Phi_i$)
        var_j: convective part ($\vec{u}_j$)

    Returns:
        List of Flux object contains divergence of var_i * var_j.
        If `var_i.type == "scalar"`, `len(list[Flux]) == 1`,
        and `var_i.type == "vector"`, `len(list[Flux]) == 3`.
    """

    def __call__(self, var_i: Field, var_j: Field) -> Any:

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
        self.flux = flux
        return self


class Laplacian(Discretizer):
    r"""Variable discretization: Laplacian

    .. math::

        \frac{\partial}{\partial x_j}
        \left(
            \Gamma^\Phi \frac{\partial \Phi}{\partial x_j}
        \right)

    Examples:

        >>> # Laplacian of scalar field
        >>> res = fvm_laplacian(coeff, phi)
        >>> res[0].x    # d^2 phi/dx_1^2
        >>> res[0].c    # d^2 phi/dx_1^2 + d^2 phi/dx_2^2 + d^2 phi/dx_3^2
        >>> # Laplacian of vector field
        >>> res = fvm_laplacian(coeff, u)
        >>> res[0].x    # d^2 u_1/dx_1^2
        >>> res[0].c    # d^2 u_1/dx_1^2 + d^2 u_1/dx_2^2 + d^2 u_1/dx_3^2

    Args:
        coeff: coefficient of the Laplacian operator ($\Gamma^\Phi$)
        var: Variable object to be discretized ($\Phi$)

    Returns:
        List of Flux object contains Laplacian of `var`.
        If `var.type == "scalar"`, `len(list[Flux]) == 1`,
        and `var.type == "vector"`, `len(list[Flux]) == 3`.

    """

    def __call__(self, coeff: float, var: Field) -> Any:

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

        self.flux = flux

        return self


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
