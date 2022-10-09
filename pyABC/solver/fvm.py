#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM) """
import warnings
from typing import Any
from typing import Optional

import torch

from pyABC.core.fields import Variables
from pyABC.solver.fluxes import DimOrder
from pyABC.solver.fluxes import FaceDir
from pyABC.solver.fluxes import Flux
from pyABC.solver.fluxes import NormalDir

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

    def __call__(self, var: Variables) -> Any:

        # Create empty flux
        self.flux = [Flux(var.NX, var.DX)]

        return self


class Grad(Discretizer):
    r"""Finite volume operator for the gradient.
    Args:
        var: Variables to be solved ($\Phi$)
    """

    def __call__(self, var: Variables) -> Any:

        self.flux = fvm_grad(var)

        return self.flux


class Div(Discretizer):
    """Divergence"""

    def __call__(self, var_i: Variables, var_j: Variables) -> list[Flux]:

        self.flux = fvm_div(var_i, var_j)

        return self.flux


class Laplacian(Discretizer):
    """Laplacian"""

    def __call__(self, coeffs: float, var: Variables) -> list[Flux]:

        self.flux = fvm_laplacian(coeffs, var)

        return self.flux


def fvm_grad(var: Variables) -> list[Flux]:
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

    for i in range(var.dim):

        flux = Flux(var.NX, var.DX)

        if var.type == "scalar":
            f_var = var()
        else:
            f_var = var()[i]

        flux = _flux_linear(flux, f_var)
        flux = _fvm_i_bc_apply(var, flux, "grad")
        flux.set_vec()
        f_return.append(flux)

    return f_return


def fvm_div(var_i: Variables, var_j: Variables) -> list[Flux]:
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

    # Dimension check for the input Variables.
    if var_i != var_j:

        msg = (
            "Two variables have different dimensions!\n"
            f"- var_i: NX({var_i.NX}), DX({var_i.DX})\n"
            f"- var_j: NX({var_j.NX}), DX({var_j.DX}) "
        )

        raise ValueError(msg)

    f_return = []

    for i in range(var_i.dim):

        f_set = []
        for j in range(var_j.dim):

            flux = Flux(var_i.NX, var_i.DX)

            # Calculate field value at the cell surface,
            # and var_j is always a vector (velocity)
            if var_i.type == "scalar":
                # Convective flux: U * Phi
                f_var = var_j()[j] * var_i()
            else:
                # Convective flux: U * U
                f_var = var_j()[j] * var_i()[i]

            # Interpolate node value to the cell surface
            flux = _flux_linear(flux, f_var)
            # BCs
            flux = _fvm_ij_bc_apply(var_i, var_j, flux)
            # Get x, y, z flux
            flux.set_vec()

            f_set.append(flux)

        # Append in i-direction
        f = _assign_flux(f_set)
        f.sum()
        f_return.append(f)

    return f_return


def _assign_flux(f_set: list[Flux]) -> Flux:
    """Add all flux stored in the list."""

    flux_sum = f_set[0].copy_reset()

    flux_sum.x = f_set[0].x
    flux_sum.y = f_set[1].y
    flux_sum.z = f_set[2].z

    return flux_sum


def fvm_laplacian(coeff: float, var: Variables) -> list[Flux]:
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

    # Construct grad operator
    grad = Grad()

    f_return = []

    for dir in range(var.dim):

        f_set = []

        for j_dir in DIR:

            flux = Flux(var.NX, var.DX)

            # Gradient
            grad_j = getattr(grad(var)[dir], j_dir)

            flux = _flux_linear(flux, grad_j)

            flux = _fvm_i_bc_apply(var, flux, "laplacian")
            flux *= coeff
            flux.set_vec()
            f_set.append(flux)

        f = _assign_flux(f_set)
        f.sum()
        f_return.append(f)

    return f_return


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


def _fvm_i_bc_apply(var: Variables, flux: Flux, type: str = "grad") -> Flux:
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


def _fvm_ij_bc_apply(
    var_i: Variables,
    var_j: Variables,
    flux: Flux,
) -> Flux:
    """Apply BC for the product of two variables.

    Args:
        var_i: target variable. Same Variable type with the final return type
        var_j: the variable represents the advection velocity
        flux: flux calculated from nodes

    Returns:
        Boundary assigned Flux
    """

    if var_i.bcs is None or var_j.bcs is None:
        _is_bcs_empty(var_i.bcs, var_j.bcs)
    else:
        _bcs_len_check(var_i.bcs, var_j.bcs)

        # Apply Bcs <<<<<<<<<<<<<< THIS IS WRONG!!!!
        for id, (bc_i, bc_j) in enumerate(zip(var_i.bcs, var_j.bcs)):

            # var_i is always the referen!
            face = bc_i.bc_face
            face_dir = getattr(FaceDir, face)
            face_normal = flux.normal[face_dir.value]
            mask = var_i.masks[id]

            flux_at_bc_i = torch.zeros_like(getattr(flux, face))
            flux_at_bc_j = torch.zeros_like(getattr(flux, face))

            flux_for_i = bc_i.apply(
                mask,
                flux_at_bc_i,
                flux.surface[face_dir.value] * face_normal,
                flux.vol,
                var_i(),
            )

            flux_for_j = bc_j.apply(
                mask,
                flux_at_bc_j,
                flux.surface[face_dir.value] * face_normal,
                flux.vol,
                var_j(),
            )

            getattr(flux, face)[mask] = flux_for_i[mask] * flux_for_j[mask]

    return flux


def _bcs_len_check(bc_i: list, bc_j: list) -> None:
    """Check bcs length. If two bcs have different lengths, raise
    SizeDoesNotMatchError.

    Args:
        bc_i: list of boundary objects for the major variable
        bc_j: list of boundary objects for the minor variable

    Raises:
        if length of two bcs list is not matched: :mod:`SizeDoesNotMatchError`
    """

    if len(bc_i) != len(bc_j):
        from pyABC.tools.errors import SizeDoesNotMatchError

        msg = (
            "Two variables have different number of BCs!"
            f"len(var_i.bcs): {len(bc_i)}, len(var_j.bcs): {len(bc_j)}"
        )
        raise SizeDoesNotMatchError(msg)
    else:
        pass


def _is_bcs_empty(bc_i: Optional[list], bc_j: Optional[list]) -> bool:
    """Check whether both bcs are empty or not. If only one of bcs is
    None, raise warning but do nothing else.

    Args:
        bc_i: list of boundary objects for the major variable
        bc_j: list of boundary objects for the minor variable

    Returns:
        Bool. If any of lists of the bcs is empty, return True.
    """

    status = False

    if bc_i is None and bc_j is None:
        status = True
    else:
        from pyABC.tools.errors import WrongInputWarning

        msg = (
            "One variable has no BCs! Assume that to be None."
            f"len(var_i.bcs): {None if bc_i is None else len(bc_i)}, "
            f"len(var_j.bcs): {None if bc_j is None else len(bc_j)}"
        )

        warnings.warn(msg, WrongInputWarning)
        status = True

    return status
