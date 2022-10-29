#!/usr/bin/env python3
"""New FVM module. WIP"""
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyapes.core.geometry.basis import DIR
from pyapes.core.variables import Flux
from pyapes.core.variables import Field
from pyapes.core.solver.fvc import Grad


@dataclass(eq=False)
class Discretizer(ABC):
    """Base class of FVM discretization.

    Examples:

        >>> # Laplacian of scalar field same for Div
        >>> laplacian = Laplacian()
        >>> res = laplacian(coeff, phi)
        >>> res.flux(0, "xl")    # d^2 phi/dx_1^2 on the left side of x directional cell face
        >>> res.flux_sum()
        >>> res.flux(0, "x")     # averaged cell centered value in x
    """

    # Init relavent attributes
    _ops: dict[int, dict[str, Union[Callable, str]]] = field(
        default_factory=dict
    )
    _rhs: Optional[Tensor] = None

    @abstractproperty
    def ops(self) -> dict[int, dict[str, Union[Callable, str]]]:
        ...

    @abstractproperty
    def rhs(self) -> Optional[Tensor]:
        ...

    @abstractproperty
    def var(self) -> Field:
        ...

    @abstractproperty
    def flux(self) -> Flux:
        ...

    @staticmethod
    @abstractmethod
    def Aop(*args, **kwargs) -> Tensor:
        """Obtain operation matrix to solve the linear system."""
        ...

    def set_config(self, config: dict):

        self.config = config

    def __eq__(self, other: Union[Tensor, float]) -> Any:

        if isinstance(other, Tensor):
            self._rhs = other
        else:
            self._rhs = torch.zeros_like(self.var()) + other

        return self

    def __add__(self, other: Any) -> Any:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update(
            {idx[-1] + 1: {"flux": other.flux, "op": other.__class__.__name__}}
        )

        return self

    def __sub__(self, other: Any) -> Any:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update(
            {
                idx[-1]
                + 1: {"flux": other.flux * -1, "op": other.__class__.__name__}
            }
        )

        return self


@dataclass(eq=False)
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

    _var: Optional[Field] = None
    _flux: Optional[Flux] = None

    def __call__(self, coeff: float, var: Field) -> Any:

        self._coeff = coeff
        self._var = var

        # Need to store Callable Aop, Var, coeffs so on...
        # So that I can use them in pyapes.core.solver.linalg.solve
        # by looping over self._ops to construct Aop
        self._ops[0] = {"Aop": Laplacian.Aop, "op": self.__class__.__name__}

        return self

    @property
    def coeff(self) -> float:
        return self._coeff

    @property
    def var(self) -> Optional[Field]:
        return self._var

    @property
    def flux(self) -> Optional[Flux]:
        return self._flux

    @staticmethod
    def Aop(coeff: float, var: Field) -> Flux:

        area = var.mesh.A
        vol = var.mesh.V

        grad = Grad(var)
        flux = Flux()

        for i in range(var.dim):
            for j in range(var.dim):

                # L is negative since wall normal vector is opposite direction of coordinate
                fl = -(
                    (
                        (torch.roll(grad(0, DIR[j]), 1, j) + grad(0, DIR[j]))
                        / (2 * var.dx[j])
                        * coeff
                    )
                    * area[DIR[j] + "l"]
                    / vol
                )
                flux.to_face(i, DIR[j], "l", fl)
                fr = (
                    (
                        (torch.roll(grad(0, DIR[j]), -1, j) + grad(0, DIR[j]))
                        / (2 * var.dx[j])
                        * coeff
                    )
                    * area[DIR[j] + "r"]
                    / vol
                )
                flux.to_face(i, DIR[j], "r", fr)

        return flux
