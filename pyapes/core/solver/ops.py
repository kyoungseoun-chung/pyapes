#!/usr/bin/env python3
"""Module for the various solvers.

Usage of fvm and fvc is a bit different than openFOAM.
Here, fvc returns dictionary of `torch.Tensor` as a result of discretization,
on the other hand, fvm returns operation matrix, `Aop` of each individual discretization scheme.

Desired usage of solver module is as follows:

    >>> fdm = FDM(config)   // initialize FDM discretization
    >>> solver = Solver(config)  // initialize solver
    >>> solver.set_eq(fdm.ddt(var_i) + ...
        fdm.div(var_i, var_j) - fdm.laplacian(c, var_i), var_i) == ...
        fdm.tensor(var) )// set PDE
    >>> solver.solve() // solve PDE

"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import TypedDict

import torch
from torch import Tensor

from pyapes.core.variables import Field


class OPStype(TypedDict):
    """Typed dict for the operation types."""

    name: str
    Aop: Callable[..., Tensor]
    inputs: tuple[float, Field] | tuple[Field, Field] | tuple[Field]
    var: Field
    sign: float | int


@dataclass(eq=False)
class Discretizer:
    """Base class of FVM discretization."""

    # Init relevant attributes
    _ops: dict[int, OPStype] = field(default_factory=dict)
    _rhs: Tensor | None = None

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

    def __eq__(self, other: Tensor | float) -> Discretizer:

        if isinstance(other, Tensor):
            self._rhs = other
        else:
            self._rhs = torch.zeros_like(self.var()) + other

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


@dataclass
class Solver:
    """`pyapes` finite volume method solver module.

    Example:
    -> Add once everything is ready.

    Args:
        config: solver configuration.
    """

    config: dict
    """Solver configuration."""

    def set_eq(self, eq: Discretizer) -> None:
        """Construct PDE to solve.
        (Actually construct Aop and RHS of the equation)

        Args:
            op: operation for the discretization
            dt_var: variable to be solved
        """

        # Target variable
        self.var = eq.var

        # Collection of discretized fluxes
        self.eqs = eq.ops
        # RHS of the equation
        self.rhs = eq.rhs

    def Aop(self) -> Tensor:
        """Return tensor of discretized operation."""

        res = torch.zeros_like(self.eqs[0]["var"]())
        for op in self.eqs:
            res += (
                self.eqs[op]["Aop"](*self.eqs[op]["inputs"])
                * self.eqs[op]["sign"]
            )

        return res

    def __call__(self) -> Field:

        assert self.var is not None, "Solver: target variable is not defined!"

        Aop = torch.zeros_like(self.var())
        for eq in self.eqs:
            pass

        raise NotImplementedError
