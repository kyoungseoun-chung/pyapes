#!/usr/bin/env python3
"""Module for the various solvers.

Usage of fvm and fvc is a bit different than openFOAM.
Here, fvc returns dictionary of `torch.Tensor` as a result of discretization,
on the other hand, fvm returns operation matrix, `Aop` of each individual discretization scheme.

Desired usage of solver module is as follows:
Solver and FDM have separate configurations.

    >>> fdm = FDM(config)   // initialize FDM discretization
    >>> solver = Solver(config)  // initialize solver
    >>> solver.set_eq(fdm.ddt(var_i) + ...
        fdm.div(var_i, var_j) - fdm.laplacian(c, var_i), var_i) == ...
        fdm.tensor(var) )// set PDE
    >>> solver.solve() // solve PDE

"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pyapes.core.solver.fdm import Discretizer
from pyapes.core.variables import Field


@dataclass(repr=False)
class Solver:
    """`pyapes` finite volume method solver module.

    Example:
    -> Add once everything is ready.

    Args:
        config: solver configuration.
    """

    config: dict[str, dict[str, str | float | int | bool]] | None = None
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
        """Return tensor of discretized operation used for the Conjugated gradient method.
        Therefore, from the system of equation `Ax = b`, Aop will be `-Ax`.
        """

        res = torch.zeros_like(self.var())

        for op in self.eqs:

            if self.eqs[op]["name"].lower() == "ddt":
                continue
            elif op > 1 and self.eqs[op]["name"].lower() == "ddt":
                raise ValueError(
                    "FDM: ddt is not allowed in the middle of the equation!"
                )

            res += (
                self.eqs[op]["Aop"](*self.eqs[op]["inputs"])
                * self.eqs[op]["sign"]
            )

        if self.eqs[0]["name"].lower() == "ddt":
            assert self.eqs[0]["other"] is not None, "FDM: dt is not defined!"

            if self.rhs is not None:
                self.rhs *= self.eqs[0]["other"]["dt"]

            res *= self.eqs[0]["other"]["dt"]
            res += self.eqs[0]["Aop"](*self.eqs[0]["inputs"])

        return res

    def __repr__(self) -> str:
        desc = ""
        for i, op in enumerate(self.eqs):
            desc += f"{i} - {self.eqs[op]['name']}, input: {self.eqs[op]['inputs']}\n"

        desc += f"{len(self.eqs)+1} - RHS, input: {self.rhs}\n"
        return desc

    def __call__(self) -> Field:

        assert self.var is not None, "Solver: target variable is not defined!"

        Aop = torch.zeros_like(self.var())
        for eq in self.eqs:
            pass

        raise NotImplementedError
