#!/usr/bin/env python3
"""OpenFoam inspired FDM solver.

Usage of `fdm` and `fdc` is resemble to `fvm` and `fvc` in the OpenFoam.
Here, `fdc` returns an explicit discretization of `Field` variable,
on the other hand, `fdm` returns operation matrix, `Aop` of each discretization scheme.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from pyapes.core.solver.fdm import Discretizer
from pyapes.core.solver.linalg import solve
from pyapes.core.variables import Field


@dataclass(repr=False)
class Solver:
    """`pyapes` finite volume method solver module.

    Note:
        - Solver and FDM have separate configurations.

    Example:

        >>> fdm = FDM(config)   // initialize FDM discretization
        >>> solver = Solver(config)  // initialize solver
        >>> solver.set_eq(fdm.ddt(var_i) + ...
            fdm.div(var_i, var_j) - fdm.laplacian(c, var_i), var_i) == ...
            fdm.tensor(var) )// set PDE
        >>> solver.solve() // solve PDE

    Args:
        config: solver configuration. Contains solver method, tolerance, max iteration, etc.
    """

    config: None | (dict[str, dict[str, str | float | int | bool]]) = None
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

        # Restting ops and rhs to avoid unnecessary copy when fdm is used multiple times in separate solvers
        eq.ops = {}
        eq.rhs = None

    def Aop(self, target: Field) -> Tensor:
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
                self.eqs[op]["Aop"](*self.eqs[op]["param"], target)
                * self.eqs[op]["sign"]
            )

        if self.eqs[0]["name"].lower() == "ddt":
            assert self.eqs[0]["other"] is not None, "FDM: dt is not defined!"

            if self.rhs is not None:
                self.rhs *= self.eqs[0]["other"]["dt"]

            res *= self.eqs[0]["other"]["dt"]
            res += self.eqs[0]["Aop"](*self.eqs[0]["param"], target)

        return res

    def solve(self) -> dict[str, int | float | bool]:
        """Solve the PDE."""

        assert (
            self.var is not None and self.rhs is not None
        ), "Solver: target variable or rhs is missing. Did't you forget to set equation?"

        assert self.config is not None, "Solver: config is missing!"

        report = solve(
            self.var, self.rhs, self.Aop, self.config["fdm"], self.var.mesh
        )

        return report

    def __repr__(self) -> str:
        desc = ""
        for op in self.eqs:
            desc += f"{op} - {self.eqs[op]['name']}, target: {self.eqs[op]['target']}, param: {self.eqs[op]['param']}\n"

        desc += f"{len(self.eqs)+1} - RHS, input: {self.rhs}\n"
        return desc
