#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Optional
from typing import TypedDict

import torch
from torch import Tensor

from pyapes.core.solver.fdc import FDC
from pyapes.core.variables import Field


class OPStype(TypedDict):
    """Typed dict for the operation types."""

    name: str
    Aop: Callable[..., Tensor]
    inputs: tuple[float, Field] | tuple[Field, Field, dict] | tuple[Field]
    sign: float | int
    other: dict[str, float] | None


@dataclass(eq=False)
class Discretizer:
    """Base class of FVM discretization."""

    # Init relevant attributes
    _ops: dict[int, OPStype] = field(default_factory=dict)
    _rhs: Tensor | None = None
    _config: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def ops(self) -> dict[int, OPStype]:
        """Collection of operators used in `pyapes.core.solver.Solver().set_eq()`"""
        return self._ops

    @property
    def rhs(self) -> Tensor | None:
        """RHS of `set_eq()`"""
        return self._rhs

    @property
    def var(self) -> Field:
        """Primary Field variable to be discretized."""
        raise NotImplementedError

    def update_config(self, config: dict[str, dict[str, str]]) -> None:
        """Update solver configuration.

        Args:
            config: solver configuration.
        """
        self._config = config

    @property
    def config(self) -> dict[str, dict[str, str]]:
        return self._config

    def __eq__(self, other: Field | Tensor | float) -> Discretizer:

        if isinstance(other, Tensor):
            self._rhs = other
        elif isinstance(other, Field):
            self._rhs = other()
        else:
            self._rhs = torch.zeros_like(self.var()) + other

        assert (
            self._rhs.shape == self.var().shape
        ), f"Discretizer: RHS shape {self._rhs.shape} does not match {self.var().shape}!"

        return self

    def __add__(self, other: Discretizer) -> Discretizer:

        idx = list(self._ops.keys())
        self._ops.update({idx[-1] + 1: other.ops[0]})

        return self

    def __sub__(self, other: Discretizer) -> Discretizer:

        idx = list(self._ops.keys())
        other.ops[0]["sign"] = -1
        self._ops.update({idx[-1] + 1: other.ops[0]})

        return self


@dataclass(eq=False)
class Grad(Discretizer):
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Args:
        var: Field object to be discretized ($\Phi$).

    """

    def __call__(self, var: Field) -> Grad:

        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "inputs": (var,),
            "sign": 1.0,
            "other": None,
        }
        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(var: Field) -> Tensor:

        fdc = FDC()

        return fdc.grad(var)


# Ddt is not yet functional
class Ddt(Discretizer):
    r"""Variable discretization: Time derivative.

    Note:
        - Currently only support `Euler Implicit`.
    """

    def __call__(self, var: Field) -> Ddt:

        try:
            dt = var.dt
        except AttributeError:
            raise AttributeError("FDM: No time step is specified.")

        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "inputs": (var,),
            "sign": 1.0,
            "other": {"dt": dt},
        }

        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(var: Field) -> Tensor:

        return FDC().ddt(var)


class Div(Discretizer):
    r"""Variable discretization: Divergence

    Note:
        - Currently supports `central difference`, and `upwind` schemes.

    .. math::

        \frac{\partial}{\partial x_j}
        \left(
            u_j \phi_i
        \right)

    Args:
        var_i: Field object to be discretized ($\Phi_i$)
        var_j: convective variable ($\vec{u}_j$)
    """

    def __call__(self, var_i: Field, var_j: Field) -> Div:

        self._var_i = var_i
        self._var_j = var_j
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "inputs": (var_i, var_j, self.config),
            "sign": 1.0,
            "other": None,
        }

        return self

    @property
    def var(self) -> Field:
        return self._var_i

    @staticmethod
    def Aop(
        var_i: Field, var_j: Field, config: dict[str, dict[str, str]]
    ) -> Tensor:

        # Div operator need config options
        fdc = FDC()
        fdc.update_config("div", "limiter", config["div"]["limiter"])

        return fdc.div(var_i, var_j)


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

    def __call__(self, coeff: float, var: Field) -> Laplacian:

        self._coeff = coeff
        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "inputs": (coeff, var),
            "sign": 1.0,
            "other": None,
        }

        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(gamma: float, var: Field) -> Tensor:
        return FDC().laplacian(gamma, var)


class RHS(Discretizer):
    r"""Simple interface to return RHS of PDE."""

    def __call__(self, var: Field | Tensor | float) -> Field | Tensor | float:
        return FDC().rhs(var)


class FDM:
    """Collection of the operators for finite difference discretizations.

    Spatial discretization supports for:

        * `div`: Divergence (central difference, upwind)
        * `laplacian`: Laplacian (central difference)
        * `grad`: Gradient (central difference, but poorly treats the edges. You must set proper boundary conditions.)

    `rhs` simply return `torch.Tensor`.

    And temporal discretization using Euler Implicit can be accessed via `ddt`.
    """

    div: Div = Div()
    laplacian: Laplacian = Laplacian()
    grad: Grad = Grad()
    rhs: RHS = RHS()
    ddt: Ddt = Ddt()

    def set_config(self, config: dict[str, dict[str, str]]) -> None:
        """Set the configuration options for the discretization operators.

        Args:
            config: configuration options for the discretization operators.

        Returns:
            FDM: self
        """

        self.config = config

        # Div operator requires config
        self.div.update_config(config)
