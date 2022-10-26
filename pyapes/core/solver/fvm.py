#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM) """
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyapes.core.geometry.basis import DIR
from pyapes.core.solver.fluxes import Flux
from pyapes.core.variables import Field


@dataclass(eq=False)
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

    # Init relavent attributes
    _ops: dict[int, dict[str, Union[Callable, str]]] = field(
        default_factory=dict
    )
    _rhs: Optional[Tensor] = None

    @property
    def ops(self) -> dict[int, dict[str, Union[Callable, str]]]:
        return self._ops

    @property
    def rhs(self) -> Optional[Tensor]:
        return self._rhs

    @property
    def var(self) -> Field:
        raise NotImplementedError

    @property
    def flux(self) -> Flux:
        raise NotImplementedError

    @property
    def Aop(self) -> Tensor:
        """Obtain operation matrix to solve the linear system."""
        raise NotImplementedError

    def set_config(self, config: dict):

        self.config = config

    def __eq__(self, other: Union[Tensor, float]) -> Any:

        if isinstance(other, Tensor):
            self._rhs = other
        else:
            self._rhs = torch.zeros_like(self.var()) + other

        return self

    def __add__(self, other: Any) -> Any:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update(
            {idx[-1] + 1: {"flux": other.flux, "op": other.__class__.__name__}}
        )

        return self

    def __sub__(self, other: Any) -> Any:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update(
            {
                idx[-1]
                + 1: {"flux": other.flux * -1, "op": other.__class__.__name__}
            }
        )

        return self


@dataclass(eq=False)
class Ddt(Discretizer):
    """Time discretization."""

    _var: Optional[Field] = None
    _flux: Optional[Flux] = None

    def __call__(self, var: Field) -> Any:

        self._var = var

        return self

    def euler_explicit(self) -> None:
        pass

    def crank_nicolson(self) -> None:
        pass

    @property
    def var(self) -> Optional[Field]:
        return self._var

    @property
    def flux(self) -> Optional[Flux]:
        return self._flux


@dataclass(eq=False)
class Grad(Discretizer):
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Args:
        var: Field object to be discretized ($\Phi$).

    """
    _var: Optional[Field] = None
    _flux: Optional[Flux] = None

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
    def var(self) -> Optional[Field]:
        return self._var

    @property
    def flux(self) -> Optional[Flux]:
        return self._flux


@dataclass(eq=False)
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

    _var: Optional[Field] = None
    _flux: Optional[Flux] = None

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

        self._flux = flux
        self._ops[0] = {"flux": flux, "op": self.__class__.__name__}

        return self

    @property
    def var(self) -> Optional[Field]:
        return self._var

    @property
    def flux(self) -> Optional[Flux]:
        return self._flux


@dataclass(eq=False)
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

    _var: Optional[Field] = None
    _flux: Optional[Flux] = None

    def __call__(self, coeff: float, var: Field) -> Any:

        self._coeff = coeff
        self._var = var

        # Need to store Callable Aop, Var, coeffs so on...
        # So that I can use them in pyapes.core.solver.linalg.solve
        # by looping over self._ops to construct Aop
        self._ops[0] = {"flux": self.Aop, "op": self.__class__.__name__}

        return self

    @property
    def coeff(self) -> float:
        return self._coeff

    @property
    def var(self) -> Optional[Field]:
        return self._var

    @property
    def flux(self) -> Optional[Flux]:
        return self._flux

    @staticmethod
    def Aop(coeff: float, var: Field) -> Flux:

        area = var.mesh.A
        vol = var.mesh.V

        fvm_Grad = Grad()
        grad = fvm_Grad(var)
        flux = Flux()

        for i in range(var.dim):
            for j in range(var.dim):

                # L is negative since wall normal vector is opposite direction of coordinate
                fl = -(
                    (
                        (torch.roll(grad(0, DIR[j]), 1, j) + grad(0, DIR[j]))
                        / (2 * var.dx[j])
                        * coeff
                    )
                    * area[DIR[j] + "l"]
                    / vol
                )
                flux.to_face(i, DIR[j], "l", fl)
                fr = (
                    (
                        (torch.roll(grad(0, DIR[j]), -1, j) + grad(0, DIR[j]))
                        / (2 * var.dx[j])
                        * coeff
                    )
                    * area[DIR[j] + "r"]
                    / vol
                )
                flux.to_face(i, DIR[j], "r", fr)

        return flux


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
