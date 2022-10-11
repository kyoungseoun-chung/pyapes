#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM) """
from typing import Any

import torch

from pyABC.core.geometry.basis import DimOrder
from pyABC.core.geometry.basis import FaceDir
from pyABC.core.geometry.basis import NormalDir
from pyABC.core.solver.fluxes import Flux
from pyABC.core.variables import Field

DIR = ["x", "y", "z"]


class Discretizer:
    """Base class of FVM discretization.

    Args:
        flux: flux object as a result of FVM operation. `len(flux)` corresponding to
              the dimension of the variable to be discretized.
        flux_total: will be used for the FVM.set_eq (not yet fixed)
    """

    # Class member
    flux: list[Flux]
    flux_total: list

    def set_config(self, config: dict):

        self.config = config

    # <<<<<<<<<<<<<<<<<<< NOT SURE <<<<<<<<<<<<<<<<<<<<<<<<
    def __add__(self, op: Any) -> Any:

        self.flux_total.append(op.flux)

        return self

    def __sub__(self, op: Any) -> Any:

        self.flux_total.append(-op.flux)

        return self

    # <<<<<<<<<<<<<<<<<<< NOT SURE <<<<<<<<<<<<<<<<<<<<<<<<


class Ddt(Discretizer):
    """Time discretization."""

    def __call__(self, var: Field) -> Any:

        raise NotImplementedError


class Grad(Discretizer):
    r"""Finite volume operator for the gradient.
    Args:
        var: Variables to be solved ($\Phi$)
    """

    def __call__(self, var: Field) -> Any:

        self.flux = fvm_grad(var)

        return self.flux


class Div(Discretizer):
    """Divergence"""

    def __call__(self, var_i: Field, var_j: Field) -> list[Flux]:

        self.flux = fvm_div(var_i, var_j)

        return self.flux


class Laplacian(Discretizer):
    """Laplacian"""

    def __call__(self, coeffs: float, var: Field) -> list[Flux]:

        self.flux = fvm_laplacian(coeffs, var)

        return self.flux


def fvm_grad(var: Field) -> list[Flux]:
    r"""Variable discretization: Gradient

    .. math::

        \frac{\partial \Phi}{\partial x_j} = \frac{1}{V_C}\sum \Phi_f \vec{n}_f \vec{S}_f

    Examples:

        >>> # Gradient of scalar field
        >>> res = fvm_grad(phi)
        >>> res[0].x    # dphi/dx_1
        >>> # Gradient of vector field
        >>> res = fvm_grad(u)
        >>> res[0].x    # du_1/dx_1

    Args:
        var: Variable object to be discretized ($\Phi$).
            - If var.bcs is None, Grad operation is done by finite difference method.

    Returns:
        List of Flux object contains gradient of `var`.
        If `var.type == "scalar"`, `len(list[Flux]) == 1`,
        and `var.type == "vector"`, `len(list[Flux]) == 3`.

    """

    f_return = []

    raise NotImplementedError


def fvm_div(var_i: Field, var_j: Field) -> list[Flux]:
    r"""Divergence operator.

    .. math::

        \frac{\partial}{\partial x_j}
        \left(
            u_j \phi_i
        \right)

    Examples:

        >>> # Advection of scalar field
        >>> res = fvm_div(phi, u)
        >>> res[0].x    # dphi/dx_1
        >>> res[0].c    # dphi/dx_1 + dphi/dx_2 + dphi/dx_3
        >>> # Advection of vector field
        >>> res = fvm_div(u, u)
        >>> res[0].x    # du_1/dx_1
        >>> res[0].c    # du_1/dx_1 + du_1/dx_2 + du_1/dx_3

    Args:
        var_i: Variable object to be discretized ($\Phi_i$)
        var_j: convective part ($\vec{u}_j$)

    Returns:
        List of Flux object contains divergence of var_i * var_j.
        If `var_i.type == "scalar"`, `len(list[Flux]) == 1`,
        and `var_i.type == "vector"`, `len(list[Flux]) == 3`.

    Raises:
        SizeDoesNotMatchError: if :attr:`var_i.NX, var_i.DX` and
        :attr:`var_j.NX, var_j.DX` are not equal.
    """

    raise NotImplementedError


def _assign_flux(f_set: list[Flux]) -> Flux:
    """Add all flux stored in the list."""

    flux_sum = f_set[0].copy_reset()

    flux_sum.x = f_set[0].x
    flux_sum.y = f_set[1].y
    flux_sum.z = f_set[2].z

    return flux_sum


def fvm_laplacian(coeff: float, var: Field) -> list[Flux]:
    r"""Variable discretization: Laplacian

    .. math::

        \frac{\partial}{\partial x_j}
        \left(
            \Gamma^\Phi \frac{\partial \Phi}{\partial x_j}
        \right)

    Examples:

        >>> # Laplacian of scalar field
        >>> res = fvm_laplacian(coeff, phi)
        >>> res[0].x    # d^2 phi/dx_1^2
        >>> res[0].c    # d^2 phi/dx_1^2 + d^2 phi/dx_2^2 + d^2 phi/dx_3^2
        >>> # Laplacian of vector field
        >>> res = fvm_laplacian(coeff, u)
        >>> res[0].x    # d^2 u_1/dx_1^2
        >>> res[0].c    # d^2 u_1/dx_1^2 + d^2 u_1/dx_2^2 + d^2 u_1/dx_3^2

    Args:
        coeff: coefficient of the Laplacian operator ($\Gamma^\Phi$)
        var: Variable object to be discretized ($\Phi$)

    Returns:
        List of Flux object contains Laplacian of `var`.
        If `var.type == "scalar"`, `len(list[Flux]) == 1`,
        and `var.type == "vector"`, `len(list[Flux]) == 3`.

    """

    raise NotImplementedError


def _flux_linear(flux: Flux, f_var: torch.Tensor) -> Flux:
    r"""Linear interpolation of the flux from the node values.

    .. math:

        \Phi_f = \frac{\Phi^{+1} + \Phi^{c}}{2}

    Args:
        flux: flux object to be calculated
        f_var: field values at the cell center
    """

    for f in FaceDir:

        flux.faces[f.value] = (
            0.5
            * (f_var + torch.roll(f_var, -NormalDir(f.name), DimOrder(f.name)))
            * flux.surface[f.value]
            * flux.normal[f.value]
            / flux.vol
        )

    return flux


def _fvm_i_bc_apply(var: Field, flux: Flux, type: str = "grad") -> Flux:
    """Apply BC for the product of a single variable.

    Args:
        var: target variable. Same Variable type with the final return type
        flux: flux calculated from nodes

    Returns:
        Boundary assigned Flux
    """

    if var.bcs is None:
        # Assume periodic
        pass
    else:
        # Apply Bcs
        for id, bc in enumerate(var.bcs):

            face_dir = getattr(FaceDir, bc.bc_face)
            flux_at_bc = getattr(flux, bc.bc_face)
            mask_at_bc = var.masks[id]

            bc.apply(
                mask_at_bc,
                flux_at_bc,
                flux.surface[face_dir.value],
                flux.vol,
                var(),
                type,
            )

    return flux
