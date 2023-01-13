#!/usr/bin/env python3
"""Module contains interface for FDM discretization.

* Each discretization class (`div`, `laplacian`, `ddt`, etc.) calls 'FDC` class. Therefore, all discretization schemes only work for `cg` related solver where `Aop` is equivalent to the discretization of target variable in target PDE.

Note:
    - I might need to integrate custom operator. (see `pystops.math.corrector`)
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import TypedDict

import torch
from torch import Tensor

from pyapes.core.solver.fdc import FDC
from pyapes.core.variables import Field


class OPStype(TypedDict):
    """Typed dict for the operation types."""

    name: str
    """Operator names"""
    Aop: Callable[..., Tensor]
    """Linear system operator. `Aop` is equivalent to `Ax` in `Ax = b`."""
    target: Field
    """Target field to be discretized."""
    param: tuple[float] | tuple[Field, dict[str, dict[str, str]]] | tuple[None]
    """Additional parameters other than target. e.g. `coeff` in `laplacian(coeff, var)`."""
    sign: float | int
    """Sign to be applied."""
    other: dict[str, float] | None
    """Additional information. e.g. `dt` in `Ddt`."""


@dataclass(eq=False)
class Discretizer:
    """Base class of FDM discretization.

    * `__eq__` method assign RHS of the equation.
    * `__add__` and `__sub__` method add discretization of the equation.
        * Unlike `__add__`, `__sub__` method updates the sign of the discretization. And later applied when `Solver` class calls `Aop` method.
    """

    # Init relevant attributes
    _ops: dict[int, OPStype] = field(default_factory=dict)
    _rhs: Tensor | None = None
    _config: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def ops(self) -> dict[int, OPStype]:
        """Collection of operators used in `pyapes.core.solver.Solver().set_eq()`"""
        return self._ops

    @ops.setter
    def ops(self, other: dict) -> None:
        self._ops = other

    @property
    def rhs(self) -> Tensor | None:
        """RHS of `set_eq()`"""
        return self._rhs

    @rhs.setter
    def rhs(self, other: Tensor | None) -> None:
        self._rhs = other

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

    def __neg__(self) -> Discretizer:

        self._ops[0]["sign"] = -1

        return self


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
            "target": var,
            "param": (None,),
            "sign": 1.0,
            "other": None,
        }
        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(_, var: Field) -> Tensor:

        fdc = FDC()

        return fdc.grad(var)


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
            "target": var,
            "param": (None,),
            "sign": 1.0,
            "other": {"dt": dt},
        }

        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(_, var: Field) -> Tensor:

        return FDC().ddt(var)


class Div(Discretizer):
    r"""Variable discretization: Divergence

    Note:
        - Currently supports `central difference`, and `upwind` schemes. Therefore, `self.config` must be defined prior to calling `Div()` and `self.config` must contain `scheme` key either `upwind` or `none`.
        - `quick` scheme is not yet supported but work in progress.

    .. math::

        \frac{\partial}{\partial x_j}
        \left(
            u_j \phi_i
        \right)

    Args:
        var_j: Field object to be discretized ($\Phi_i$)
        var_i: convective variable ($\vec{u}_j$)
    """

    def __call__(self, var_j: Field, var_i: Field) -> Div:

        self._var_i = var_i
        self._var_j = var_j
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "target": var_i,
            "param": (var_j, self.config),
            "sign": 1.0,
            "other": None,
        }

        return self

    @property
    def var(self) -> Field:
        return self._var_i

    @staticmethod
    def Aop(
        var_j: Field, config: dict[str, dict[str, str]], var_i: Field
    ) -> Tensor:

        # Div operator need config options
        fdc = FDC()
        fdc.update_config("div", "limiter", config["div"]["limiter"])

        return fdc.div(var_j, var_i)


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
            "target": var,
            "param": (coeff,),
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

    Note:
        * Spatial discretization supports for:
            * `div`: Divergence (central difference, upwind)
            * `laplacian`: Laplacian (central difference)
            * `grad`: Gradient (central difference, but poorly treats the edges. You must set proper boundary conditions.)
        * `rhs` simply return `torch.Tensor`.
        * And temporal discretization using Euler Implicit can be accessed via `ddt`.

    """

    div: Div = Div()
    """Divergence operator: `div(var_j, var_i)`."""
    laplacian: Laplacian = Laplacian()
    """Laplacian operator: `laplacian(coeff, var)`."""
    grad: Grad = Grad()
    """Gradient operator: `grad(var)`."""
    rhs: RHS = RHS()
    """Assign RHS of PDE: `rhs(var)`."""
    ddt: Ddt = Ddt()
    """Time discretization: `ddt(var)`."""

    def __init__(
        self, config: dict[str, dict[str, str]] | None = None
    ) -> None:
        """Initialize FDM. If `config` is provided, `config` will be set via `self.set_config` function.

        Args:
            config (Optional[dict[str, dict[str, str]]]): configuration options for the discretization operators.
        """

        if config is not None:
            self.set_config(config)

    def set_config(self, config: dict[str, dict[str, str]]) -> None:
        """Set the configuration options for the discretization operators.

        Args:
            config: configuration options for the discretization operators.

        Returns:
            FDM: self
        """

        self.config = config

        # Currently only `Div`` operator requires config
        self.div.update_config(config)
