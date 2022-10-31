#!/usr/bin/env python3
"""Module for the various solvers.

Usage of fvm and fvc is a bit different than openFOAM.
Here, fvc returns dictionary of `torch.Tensor` as a result of discretization,
on the other hand, fvm returns operation matrix, `Aop` of each individual discretization scheme.

"""
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from pyapes.core.solver.fvc import FTensor as FVC_Tensor
from pyapes.core.solver.fvc import FVC_type
from pyapes.core.solver.fvc import Grad as FVC_Grad
from pyapes.core.solver.fvc import Laplacian as FVC_Laplacian
from pyapes.core.solver.fvc import Source as FVC_Source
from pyapes.core.solver.fvm import Ddt as FVM_Ddt
from pyapes.core.solver.fvm import Div as FVM_Div
from pyapes.core.solver.fvm import FVM_type
from pyapes.core.solver.fvm import Grad as FVM_Grad
from pyapes.core.solver.fvm import Laplacian as FVM_Laplacian
from pyapes.core.variables import Field


@dataclass
class FVC:
    """Collection of the operators for explicit finite volume discretizations."""

    grad: FVC_Grad = FVC_Grad()
    laplacian: FVC_Laplacian = FVC_Laplacian()
    source: FVC_Source = FVC_Source()
    tensor: FVC_Tensor = FVC_Tensor()


@dataclass
class FVM:
    """Collection of the operators for implicit finite volume discretizations."" """

    ddt: FVM_Ddt = FVM_Ddt()
    grad: FVM_Grad = FVM_Grad()
    div: FVM_Div = FVM_Div()
    laplacian: FVM_Laplacian = FVM_Laplacian()


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

    def set_eq(self, eq: Union[FVC_type, FVM_type]) -> None:
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
        pass

    # NOTE: Will be structured like this but not yet validated.
    def Aop(self, var: Field) -> Tensor:

        res = torch.zeros_like(var())
        for op in self.eqs:
            flux = self.eqs[op](var)
            res += flux.sum()

        return res

    def __call__(self) -> Field:

        assert self.var is not None, "Solver: target variable is not defined!"

        Aop = torch.zeros_like(self.var())
        for eq in self.eqs:
            if self.eqs[eq]["op"] != "Ddt":
                pass

        raise NotImplementedError
