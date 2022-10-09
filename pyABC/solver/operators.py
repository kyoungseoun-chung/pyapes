#!/usr/bin/env python3
"""Module for the various solvers.
"""
from dataclasses import dataclass
from typing import Any
from typing import Union

import torch

from pyABC.core.fields import Variables
from pyABC.solver.fdm import Laplacian as FDM_Laplacian
from pyABC.solver.fluxes import Flux
from pyABC.solver.fvm import Ddt as FVM_Ddt
from pyABC.solver.fvm import Div as FVM_Div
from pyABC.solver.fvm import Grad as FVM_Grad
from pyABC.solver.fvm import Laplacian as FVM_Laplacian


@dataclass
class FVC:
    """Collection of the operators based on volume node discretization.

    Args:
        laplacian: Laplacian operator. Used to solve the Poisson equation.
    """

    laplacian: FDM_Laplacian = FDM_Laplacian()

    def solve(self, op: Variables) -> Variables:
        """This do nothing. Just to avoid any confusion.

        For example:
            >>> # Belows are identical
            >>> fvc.solve(fvc.laplacian(var) == rhs)
            >>> (fvc.laplacian(var) == rhs)

        """

        return op


@dataclass
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

    ddt: FVM_Ddt = FVM_Ddt()
    grad: FVM_Grad = FVM_Grad()
    div: FVM_Div = FVM_Div()
    laplacian: FVM_Laplacian = FVM_Laplacian()

    def set_eq(self, eq: Flux, dt_var: Variables) -> None:
        """Construct PDE to solve.
        (Acutally just store data for future self.solve function call)

        Args:
            op: operation for the discretization
            dt_var: variable to be solved
        """

        # Variable to be solved
        self.var = dt_var

        # Discretized fluxes
        self.eq = eq

    def solve(self, op: Flux) -> Variables:

        self.var.VAR = op.sum - self.rhs

        return self.var

    def __eq__(self, rhs: Union[torch.Tensor, float]) -> Any:

        if type(rhs) == float:

            self.rhs = torch.zeros_like(self.var.VAR) + float

        self.rhs = rhs

        return self


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
