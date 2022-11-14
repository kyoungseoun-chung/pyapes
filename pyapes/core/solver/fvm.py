#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM). Unlike `fvc`, `fvm` is designed to solve the field implicitly. Therefore, it contains more complicated data structure rather than just returning `torch.Tensor`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Union

from torch import Tensor

from .fv import Discretizer
from .fvc import FVC
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

    def __call__(self, var: Field) -> Any:

        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "inputs": (var,),
            "var": var,
            "sign": 1.0,
        }
        return self

    @staticmethod
    def Aop(var: Field) -> Tensor:

        return FVC.grad(var)


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

        self._var_i = var_i
        self._var_j = var_j
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "inputs": (var_i, var_j),
            "var": var_i,
            "sign": 1.0,
        }

        return self

    @staticmethod
    def Aop(var_i: Field, var_j: Field) -> Tensor:

        return FVC.div(var_i, var_j)


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
            "inputs": (coeff, var),
            "var": var,
            "sign": 1.0,
        }

        return self

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
