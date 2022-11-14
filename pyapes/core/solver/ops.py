#!/usr/bin/env python3
"""Module for the various solvers.

Usage of fvm and fvc is a bit different than openFOAM.
Here, fvc returns dictionary of `torch.Tensor` as a result of discretization,
on the other hand, fvm returns operation matrix, `Aop` of each individual discretization scheme.

"""
from dataclasses import dataclass

import torch
from torch import Tensor

from pyapes.core.solver.fv import Discretizer
from pyapes.core.solver.fvc import FVC
from pyapes.core.solver.fvm import FVM
from pyapes.core.variables import Field


@dataclass
class Solver:
    """`pyapes` finite volume method solver module.

    Example:

        >>> solver = Solver(config)
        >>> fvm = solver.fvm
        >>> fvc = solver.fvc
        >>> solver.set_eq(fvm.grad(var) + fvm.div(var, var) - fvm.laplacian(c, var), var)
        >>> fvc.solve(fvc.laplacian(P) == rhs)
        >>> var = fvm.solve(fvm.eq == -P)

    Args:
        config: solver configuration.
        fvc: Volume node based discretization.
    """

    config: dict
    fvc: FVC = FVC()
    fvm: FVM = FVM()

    def set_eq(self, eq: Discretizer) -> None:
        """Construct PDE to solve.
        (Actually just store data for future self.solve function call)

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
