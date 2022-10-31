#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM). Unlike `fvc`, `fvm` is designed to solve the field implicitly. Therefore, it contains more complicated data structure rather than just returning `torch.Tensor`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from .fv import Discretizer
from .fvc import FVC
from pyapes.core.geometry.basis import DIR
from pyapes.core.variables import Field
from pyapes.core.variables import Flux


@dataclass(eq=False)
class Ddt(Discretizer):
    """Time discretization."""

    _var: Field | None = None
    _flux: Flux | None = None

    def __call__(self, var: Field) -> Any:

        self._var = var

        return self

    def euler_explicit(self) -> None:
        pass

    def crank_nicolson(self) -> None:
        pass

    @property
    def var(self) -> Field | None:
        return self._var

    @property
    def flux(self) -> Flux | None:
        return self._flux


@dataclass(eq=False)
class Grad(Discretizer):
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Args:
        var: Field object to be discretized ($\Phi$).

    """
    _var: Field | None = None
    _flux: Flux | None = None

    def __call__(self, var: Field) -> Flux:

        self._var = var

        dx = var.mesh.dx

        grad = Flux(var.mesh)

        for i in range(var.dim):

            for j in range(var.mesh.dim):

                grad.to_center(
                    i,
                    DIR[j],
                    (torch.roll(var()[i], -1, j) - torch.roll(var()[i], 1, j))
                    / (2 * dx[j]),
                )
        self._flux = grad

        return grad

    @property
    def var(self) -> Field | None:
        return self._var

    @property
    def flux(self) -> Flux | None:
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

    _var: Field | None = None
    _flux: Flux | None = None

    def __call__(self, var_i: Field, var_j: Field) -> Any:

        self._var = var_i

        flux = Flux(var_i.mesh)

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
    def var(self) -> Field | None:
        return self._var

    @property
    def flux(self) -> Flux | None:
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

    _var: Field | None = None
    _flux: Flux | None = None

    def __call__(self, coeff: float, var: Field) -> Laplacian:

        self._coeff = coeff
        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "input": (coeff, var),
        }

        return self

    @property
    def coeff(self) -> float:
        return self._coeff

    @property
    def var(self) -> Field | None:
        return self._var

    @property
    def flux(self) -> Flux | None:
        return self._flux

    @staticmethod
    def Aop(gamma: float, var: Field) -> Tensor:

        return FVC.laplacian(gamma, var)


@dataclass
class FVM:
    """Collection of the operators for implicit finite volume discretizations."" """

    ddt: Ddt = Ddt()
    grad: Grad = Grad()
    div: Div = Div()
    laplacian: Laplacian = Laplacian()


FVM_type = Union[Div, Grad, Laplacian]
