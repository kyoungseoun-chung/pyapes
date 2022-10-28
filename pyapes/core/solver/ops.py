#!/usr/bin/env python3
"""Module for the various solvers.
"""
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pyapes.core.solver.fdm import Discretizer as FDM_Discretizer
from pyapes.core.solver.fdm import Laplacian as FDM_Laplacian
from pyapes.core.solver.fvm import Ddt as FVM_Ddt
from pyapes.core.solver.fvm import Discretizer
from pyapes.core.solver.fvm import Div as FVM_Div
from pyapes.core.solver.fvm import Flux
from pyapes.core.solver.fvm import Grad as FVM_Grad
from pyapes.core.solver.fvm import Laplacian as FVM_Laplacian
from pyapes.core.variables import Field


class FVC:
    """Collection of the operators based on volume node discretization.

    Args:
        laplacian: Laplacian operator. Used to solve the Poisson equation.
    """

    laplacian: FDM_Laplacian = FDM_Laplacian()

    def set_eq(self, eq: FDM_Discretizer) -> Any:
        """Assign variable to this object."""

        self.ops = eq

        return self

    def solve(self) -> Field:

        return self.ops.solve()


class FVM:
    """Collection of the operators based on FVM.

    Example:

        >>> # WARNING: Not yet fully determined...
        >>> fvm = FVM()
        >>> fvc = FVC()
        >>> fvm.set_eq(fvm.grad(var) + fvm.div(var, var) - fvm.laplacian(c, var), var)
        >>> fvc.solve(fvc.laplacian(P) == rhs)
        >>> var = fvm.solve(fvm.eq == -P)

    Args:
        grad: Gradient Loperator. Used to solve the Poisson equation.
    """

    # operations are added in dictionary. But how?
    ddt: FVM_Ddt = FVM_Ddt()
    grad: FVM_Grad = FVM_Grad()
    div: FVM_Div = FVM_Div()
    laplacian: FVM_Laplacian = FVM_Laplacian()

    def set_eq(self, eq: Discretizer) -> None:
        """Construct PDE to solve.
        (Acutally just store data for future self.solve function call)

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
        pass

    def solve(self) -> Field:

        Aop = torch.zeros_like(self.var())
        for eq in self.eqs:
            if self.eqs[eq]["op"] != "Ddt":
                pass

        raise NotImplementedError


@dataclass
class Solver:
    """Collection of all solver operators.

    Args:
        config: solver configuration.
        fvc: Volume node based discretization.
    """

    config: dict
    fvc: FVC = FVC()
    fvm: FVM = FVM()

    def __post_init__(self):
        """Initialize solver configuration."""

        try:
            self.fvc.laplacian.set_config(self.config["fvc"])
        except KeyError:
            pass

        try:
            self.fvm.ddt.set_config(self.config["fvm"])
        except KeyError:
            pass

        try:
            self.fvm.grad.set_config(self.config["fvm"])
        except KeyError:
            pass

        try:
            self.fvm.div.set_config(self.config["fvm"])
        except KeyError:
            pass

        try:
            self.fvm.laplacian.set_config(self.config["fvm"])
        except KeyError:
            pass