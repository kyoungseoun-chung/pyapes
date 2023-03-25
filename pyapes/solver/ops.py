#!/usr/bin/env python3
"""OpenFoam inspired FDM solver.

Usage of `fdm` and `fdc` is resemble to `fvm` and `fvc` in the OpenFoam.
Here, `fdc` returns an explicit discretization of `Field` variable,
on the other hand, `fdm` returns operation matrix, `Aop` of each discretization scheme.

"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pyapes.solver.fdm import Operators
from pyapes.solver.fdm import OPStype
from pyapes.solver.linalg import ReportType
from pyapes.solver.linalg import solve
from pyapes.solver.tools import SolverConfig
from pyapes.variables import Field


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

    config: None | SolverConfig = None
    """Solver configuration."""

    def set_eq(self, eq: Operators) -> None:
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

        # Adjusting RHS based on the boundary conditions
        # NOTE: Could not fix type issue here.
        if self.rhs is not None:
            for e in self.eqs:
                if self.eqs[e]["name"] == "Div":
                    param = self.eqs[e]["param"]

                    assert len(param) == 2

                    rhs_func = self.eqs[e]["adjust_rhs"]

                    self.rhs += rhs_func(param[0], self.var, param[1])  # type: ignore
                else:
                    rhs_func = self.eqs[e]["adjust_rhs"]
                    self.rhs += rhs_func(self.var)  # type: ignore

        # Resetting ops and rhs to avoid unnecessary copy when fdm is used multiple times in separate solvers
        eq.ops = {}
        eq.rhs = None

    def Aop(self, var: Field) -> Tensor:
        """Aop interface mostly for debugging."""

        assert (
            self.rhs is not None
        ), "Solver: rhs is missing. Did't you forget to set equation?"

        return _Aop(var, self.eqs)

    def solve(self) -> ReportType:
        """Solve the PDE."""

        assert (
            self.var is not None and self.rhs is not None
        ), "Solver: target variable or rhs is missing. Did't you forget to set equation?"

        assert self.config is not None, "Solver: config is missing!"

        # Iterative linalg solver
        self.report = solve(
            self.var,
            self.rhs,
            _Aop,
            self.eqs,
            self.config["fdm"],
            self.var.mesh,
        )

        return self.report

    def __repr__(self) -> str:
        desc = ""
        for op in self.eqs:
            desc += f"{op} - {self.eqs[op]['name']}, target: {self.eqs[op]['target']}, param: {self.eqs[op]['param']}\n"

        desc += f"{len(self.eqs)+1} - RHS, input: {self.rhs}\n"
        return desc


def _Aop(target: Field, eqs: dict[int, OPStype]) -> Tensor:
    """Return tensor of discretized operation used for the Conjugated gradient method.
    Therefore, from the system of equation `Ax = b`, Aop will be `-Ax`.

    Note:
        - This function is intentionally separated from `Solver` class to make the `solve` process more transparent. (`rhs` and `eqs` are explicitly passed to the function)
    """

    res = torch.zeros_like(target())

    for op in eqs:
        if eqs[op]["name"].lower() == "ddt":
            continue
        elif op > 1 and eqs[op]["name"].lower() == "ddt":
            raise ValueError("FDM: ddt is not allowed in the middle of the equation!")

        # Compute A @ x
        # NOTE: Could not fix type issue here.
        Ax = (
            eqs[op]["Aop"](*eqs[op]["param"], target, eqs[op]["A_coeffs"])  # type: ignore
            * eqs[op]["sign"]
        )  # type: ignore

        if eqs[op]["name"].lower() == "grad":
            # If operator is grad, re-shape to match the size of the target variable
            Ax = Ax.view(target.size)

        res += Ax

    if eqs[0]["name"].lower() == "ddt":
        res += eqs[0]["Aop"](*eqs[0]["param"], target)  # type: ignore

    return res
