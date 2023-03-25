#!/usr/bin/env python3
"""Module contains interface for FDM discretization.

* Each discretization class (`div`, `laplacian`, `ddt`, etc.) calls 'FDC` class. Therefore, all discretization schemes only work for `cg` related solver where `Aop` is equivalent to the discretization of target variable in target PDE.

Note:
    - I might need to integrate custom operator. (see `pystops.math.corrector`)
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from torch import Tensor

from pyapes.mesh import Mesh
from pyapes.solver.fdc import FDC
from pyapes.solver.types import DiscretizerConfigType
from pyapes.solver.types import OPStype
from pyapes.variables import Field
from pyapes.variables.bcs import BC_type


@dataclass(eq=False)
class Operators:
    """Base class of FDM operators.

    * `__eq__` method assign RHS of the equation.
    * `__add__` and `__sub__` method add discretization of the equation.
        * Unlike `__add__`, `__sub__` method updates the sign of the discretization. And later applied when `Solver` class calls `Aop` method.
    """

    # Init relevant attributes
    _ops: dict[int, OPStype] = field(default_factory=dict)
    _rhs: Tensor | None = None
    _config: DiscretizerConfigType | None = None

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

    def update_config(self, config: DiscretizerConfigType) -> None:
        """Update solver configuration.

        Args:
            config: solver configuration.
        """
        self._config = config

    @property
    def config(self) -> DiscretizerConfigType | None:
        return self._config

    def __eq__(self, other: Field | Tensor | float) -> Operators:
        if isinstance(other, Tensor):
            self._rhs = other
        elif isinstance(other, Field):
            self._rhs = other()
        else:
            self._rhs = torch.zeros_like(self.var()) + other

        assert (
            self._rhs.shape == self.var().shape
        ), f"FDM Operators: RHS shape {self._rhs.shape} does not match {self.var().shape}!"

        return self

    def __add__(self, other: Operators) -> Operators:
        idx = list(self._ops.keys())
        self._ops.update({idx[-1] + 1: other.ops[0]})

        return self

    def __sub__(self, other: Operators) -> Operators:
        idx = list(self._ops.keys())
        other.ops[0]["sign"] = -1
        self._ops.update({idx[-1] + 1: other.ops[0]})

        return self

    def __neg__(self) -> Operators:
        self._ops[0]["sign"] = -1

        return self


class Laplacian(Operators):
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

    def __call__(self, *inputs: Any) -> Laplacian:
        if len(inputs) == 2:
            assert isinstance(
                inputs[0], int | float | Tensor
            ), "FDM Laplacian: if additional parameter is provided, it must be a float or Tensor!"

            coeffs = float(inputs[0]) if isinstance(inputs[0], int) else inputs[0]
            var = inputs[1]
        elif len(inputs) == 1:
            coeffs = None
            var = inputs[0]
        else:
            raise TypeError("FDM: invalid input type!")

        A_coeffs = FDC(self.config).laplacian.build_A_coeffs(var)

        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "target": var,
            "param": (coeffs,),
            "sign": 1.0,
            "other": None,
            "A_coeffs": A_coeffs,
            "adjust_rhs": FDC.laplacian.adjust_rhs,
        }

        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(
        param: float | Tensor | None, var: Field, A_coeffs: list[list[Tensor]]
    ) -> Tensor:
        """Compute `Ax` of the linear system `Ax = b`. If param is not None, the whole operation is multiplied by param."""

        if param is None:
            return FDC.laplacian.apply(A_coeffs, var)
        else:
            return FDC.laplacian.apply(A_coeffs, var) * param


class Grad(Operators):
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Args:
        var: Field object to be discretized ($\Phi$).

    """

    def __call__(self, *inputs: Any) -> Grad:
        if len(inputs) == 2:
            assert isinstance(inputs[0], float) or isinstance(
                inputs[0], Tensor
            ), "FDM Grad: if additional parameter is provided, it must be a float or Tensor!"
            coeffs = inputs[0]
            var = inputs[1]
        elif len(inputs) == 1:
            assert isinstance(
                inputs[0], Field
            ), "FDM Grad: invalid input type! Input must be a Field."
            coeffs = None
            var = inputs[0]
        else:
            raise TypeError("FDM: invalid input type!")

        A_coeffs = FDC.grad.build_A_coeffs(var)

        self._var = var
        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "target": var,
            "param": (coeffs,),
            "sign": 1.0,
            "other": None,
            "A_coeffs": A_coeffs,
            "adjust_rhs": FDC.grad.adjust_rhs,
        }
        return self

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(
        param: float | Tensor | None, var: Field, A_coeffs: list[list[Tensor]]
    ) -> Tensor:
        """Compute `Ax` of the linear system `Ax = b`. If param is not None, the whole operation is multiplied by param."""
        if param is None:
            return FDC().grad.apply(A_coeffs, var)
        else:
            return FDC().grad.apply(A_coeffs, var) * param


class Div(Operators):
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

    def __call__(self, *inputs: Any) -> Div:
        """It is important to note that the order of args is important here. The first input is the convective variable (`var_j`), and the second input is the field to be discretized (`var_i`)."""

        if len(inputs) == 2:
            assert (
                isinstance(inputs[0], float)
                or isinstance(inputs[0], Tensor)
                or isinstance(inputs[0], Field)
            ), "FDM Grad: if additional parameter is provided, it must be a float or Tensor or Field!"
            var_j = inputs[0]
            var_i = inputs[1]
        elif len(inputs) == 1:
            var_j = 1.0
            var_i = inputs[0]
        else:
            raise TypeError("FDM: invalid input type!")

        assert isinstance(var_i, Field), "FDM Div: var_i must be a Field!"

        self._var_j = var_j
        self._var_i = var_i

        assert self.config is not None, "FDM Div: config must be provided!"

        A_coeffs = FDC.div.build_A_coeffs(var_j, var_i, self.config)

        self._ops[0] = {
            "name": self.__class__.__name__,
            "Aop": self.Aop,
            "target": var_i,
            "param": (var_j, self.config),
            "sign": 1.0,
            "other": None,
            "A_coeffs": A_coeffs,
            "adjust_rhs": FDC.div.adjust_rhs,
        }

        return self

    @property
    def var(self) -> Field:
        return self._var_i

    @staticmethod
    def Aop(
        var_j: Field | Tensor | float,
        config: DiscretizerConfigType,
        var_i: Field,
        A_coeffs: list[list[Tensor]],
    ) -> Tensor:
        """Compute `Ax` for the linear system of `Ax=b`. If `var_j` is either `Tensor` or `float`, assume that the advection term is constant. Therefore, reuse `A_coeffs`. Otherwise, update `A_coeffs` every step to compute `Ax`."""

        if isinstance(var_j, Tensor | float):
            # Reuse A_coeffs
            return FDC().div.apply(A_coeffs, var_i)
        else:
            # Update A_coeffs
            _A_coeffs = FDC.div.build_A_coeffs(var_j, var_i, config)
            return FDC().div.apply(_A_coeffs, var_i)


class Ddt(Operators):
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
        # self._ops[0] = {
        #     "name": self.__class__.__name__,
        #     "Aop": self.Aop,
        #     "target": var,
        #     "param": (dt,),
        #     "sign": 1.0,
        #     "other": None,
        #     "adjust_rhs": self.update_rhs,
        # }

        return self

    @staticmethod
    def update_rhs(rhs: Tensor, bcs: list[BC_type], mesh: Mesh) -> None:
        pass

    @property
    def var(self) -> Field:
        return self._var

    @staticmethod
    def Aop(dt: float, var: Field) -> Tensor:
        ...

        # return FDC().ddt(dt, var)


class FDM:
    """Collection of the operators for finite difference discretizations.

    Note:
        * Spatial discretization supports for:
            * `div`: Divergence (central difference, upwind)
            * `laplacian`: Laplacian (central difference)
            * `grad`: Gradient (central difference, but poorly treats the edges. You must set proper boundary conditions.)
        * And temporal discretization using Euler Implicit can be accessed via `ddt`.

    Updates:
        * 08.02.2023:
            - Removed `rhs` since you can directly assign by using `==` operator.

    """

    laplacian: Laplacian = Laplacian()
    """Laplacian operator. Returns `Tensor`:
        >>> FDM().laplacian(coeff: float | Tensor, var: Field)
        # or
        >>> FDM().laplacian(var: Field)
    """
    grad: Grad = Grad()
    """Gradient operator. Returns `Tensor`.:
        >>> FDM().grad(coeff: float | Tensor, var: Field)
        # or
        >>> FDM().grad(var: Field)
    """
    div: Div = Div()
    """Divergence operator. Returns `Tensor`:
        >>> FDM().div(var_i: Field)
        # or
        >>> FDM().div(coeff: float | Tensor, var_i: Field)
        # or
        >>> FDM().div(var_j: Field, var_i: Field)
    """

    # ddt: Ddt = Ddt()
    """Time discretization: `ddt(var)`."""

    def __init__(self, config: DiscretizerConfigType | None = None) -> None:
        """Initialize FDM. If `config` is provided, `config` will be set via `self.set_config` function.

        Args:
            config (Optional[dict[str, dict[str, str]]]): configuration options for the discretization operators.
        """

        if config is not None:
            self.config = config

            # Currently only `Div`` operator requires config
            self.div.update_config(config)
