#!/usr/bin/env python3
"""Finite Difference for the current field `FDC`. Similar to Openfoam's `FVC` class."""
import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pyapes.core.geometry.basis import NUM_TO_DIR
from pyapes.core.mesh.tools import create_pad
from pyapes.core.mesh.tools import inner_slicer
from pyapes.core.solver.tools import fill_pad
from pyapes.core.solver.tools import fill_pad_bc
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import BC


@dataclass
class Discretizer(ABC):
    """Collection of the operators for explicit finite difference discretizations."""

    A_coeffs: tuple[Tensor, Tensor, Tensor] | None = None
    """Tuple of A operation matrix coefficients."""
    rhs_adj: Tensor | None = None
    """RHS adjustment tensor."""

    @staticmethod
    @abstractmethod
    def build_A_coeffs(var: Field) -> tuple[Tensor, Tensor, Tensor]:
        """Build the operation matrix coefficients to be used for the discretization.
        `var: Field` is required due to the boundary conditions. Should always return three tensors in `Ap`, `Ac`, and `Am` order.
        """
        ...

    @staticmethod
    @abstractmethod
    def adjust_rhs(var: Field) -> Tensor:
        """Return a tensor that is used to adjust `rhs` of the PDE."""
        ...

    @abstractmethod
    def apply(self) -> Tensor:
        """Apply discretization."""
        ...

    @abstractmethod
    def __call__(self, *_) -> Tensor:
        """Conduct whole process of discretization of input Field."""
        ...


class Laplacian(Discretizer):
    """Laplacian discretizer."""

    @staticmethod
    def build_A_coeffs(var: Field) -> tuple[Tensor, Tensor, Tensor]:

        Ap = torch.ones_like(var())
        Ac = -2.0 * torch.ones_like(var())
        Am = torch.ones_like(var())

        dx = var.dx

        # Treat boundaries
        for i in range(var.dim):

            for bc in var.bcs:
                if bc.bc_type == "neumann" or bc.bc_type == "symmetry":
                    if bc.bc_n_dir < 0:
                        # At lower side
                        Ap[i][bc.bc_mask_prev] = 2 / 3
                        Ac[i][bc.bc_mask_prev] = -2 / 3
                        Am[i][bc.bc_mask_prev] = 0.0
                    else:
                        # At upper side
                        Ap[i][bc.bc_mask_prev] = 0.0
                        Ac[i][bc.bc_mask_prev] = -2 / 3
                        Am[i][bc.bc_mask_prev] = 2 / 3
                elif bc.bc_type == "periodic":
                    if bc.bc_n_dir < 0:
                        # At lower side
                        Am[i][bc.bc_mask_prev] = 0.0
                    else:
                        # At upper side
                        Ap[i][bc.bc_mask_prev] = 0.0
                else:
                    # Dirichlet BC: Do nothing

                    pass

        Ap /= dx**2
        Ac /= dx**2
        Am /= dx**2

        return Ap, Ac, Am

    @staticmethod
    def adjust_rhs(var: Field) -> Tensor:

        rhs_adj = torch.zeros_like(var())
        dx = var.dx

        # Treat boundaries
        for i in range(var.dim):

            for bc in var.bcs:

                if bc.bc_type == "neumann":
                    at_bc = _return_bc_val(bc, var, i)
                    rhs_adj -= 2 / 3 * at_bc / dx * float(bc.bc_n_dir)
                elif bc.bc_type == "periodic":
                    # NOTE: Not sure yet
                    rhs_adj += (
                        var()[i][bc.bc_mask_forward]
                        / dx**2
                        * float(bc.bc_n_dir)
                    )
                else:
                    # Dirichlet and Symmetry BC: Do nothing
                    pass

        return rhs_adj

    def apply(self, var: Field) -> Tensor:
        assert self.A_coeffs is not None, "FDC: A_coeffs is not defined!"

        discretized = torch.zeros_like(var())

        for idx in range(var.dim):
            discretized += (
                self.A_coeffs[0] * torch.roll(var()[idx], -1)
                + self.A_coeffs[1] * var()[idx]
                + self.A_coeffs[2] * torch.roll(var()[idx], 1)
            )

        return discretized

    def __call__(self, var) -> Tensor:

        if self.A_coeffs is None:
            self.A_coeffs = self.build_A_coeffs(var)

        if self.rhs_adj is None:
            self.rhs_adj = self.adjust_rhs(var)

        return self.apply(var)


class Grad(Discretizer):
    def build_A_coeffs(self, var: Field) -> tuple[Tensor, Tensor, Tensor]:
        ...

    def adjust_rhs(self, var: Field) -> Tensor:
        ...

    def __call__(self, var) -> Tensor:
        ...


class Div(Discretizer):
    def build_A_coeffs(self, var: Field) -> tuple[Tensor, Tensor, Tensor]:
        ...

    def adjust_rhs(self, var: Field) -> Tensor:
        ...

    def __call__(self, var_j: Field, var_i: Field) -> Tensor:
        ...


class Ddt(Discretizer):
    def build_A_coeffs(self, var: Field) -> tuple[Tensor, Tensor, Tensor]:
        ...

    def adjust_rhs(self, var: Field) -> Tensor:
        ...

    def __call__(self, var) -> Tensor:
        ...


def _return_bc_val(bc: BC, var: Field, dim: int) -> Tensor | float:
    """Return boundary values."""

    if callable(bc.bc_val):
        at_bc = bc.bc_val(var.mesh.grid, bc.bc_mask)
    elif isinstance(bc.bc_val, list):
        at_bc = bc.bc_val[dim]
    elif isinstance(bc.bc_val, float | int):
        at_bc = bc.bc_val
    elif bc.bc_val is None:
        at_bc = 0.0
    else:
        raise ValueError(f"Unknown boundary condition value: {bc.bc_val}")

    return at_bc


class FDC:
    """Collection of Finite Difference discretizations. The operation is explicit, therefore, all methods return a tensor."""

    config: Optional[dict[str, dict[str, str]]] = None
    """Configuration for the discretization."""
    div: Div = Div()
    """Divergence operator: `div(var_j, var_i)`."""
    laplacian: Laplacian = Laplacian()
    """Laplacian operator: `laplacian(coeff, var)`."""
    grad: Grad = Grad()
    """Gradient operator: `grad(var)`."""
    ddt: Ddt = Ddt()
    """Time discretization: `ddt(var)`. It will only adjust RHS of the PDE."""

    def update_config(self, scheme: str, target: str, val: str):
        """Update config values."""

        if self.config is not None:
            self.config[scheme][target] = val
        else:
            self.config = {scheme: {target: val}}


@dataclass
class FDC_old:
    """Collection of the operators for explicit finite difference discretizations."""

    config: Optional[dict[str, dict[str, str]]] = None

    def update_config(self, scheme: str, target: str, val: str):
        """Update config values."""

        if self.config is not None:
            self.config[scheme][target] = val
        else:
            self.config = {scheme: {target: val}}

    def ddt(self, dt: float, var: Field) -> Tensor:
        """Time derivative of a given field.

        Note:
            - If `var` does not have old `VARo`, attribute, current `VAR` will be treated as `VARo`.
        """

        ddt = []

        for i in range(var.dim):
            try:
                var_o = var.VARo[i]
            except AttributeError:
                # If not saved, use current value (treat for the first iteration)
                var_o = var()[i]
            ddt.append((var()[i] - var_o) / dt)

        return torch.stack(ddt)

    def rhs(self, var: Field | Tensor | float) -> Field | Tensor | float:
        """Simply assign a given field to RHS of PDE."""

        return var

    def div(self, var_j: Field, var_i: Field) -> Tensor:
        """Divergence of two fields.
        Note:
            - To avoid the checkerboard problem, flux limiter is used. It supports `none`, `upwind` and `quick` limiter. (here, `none` is equivalent to the second order central difference.)
        """

        if self.config is not None and "limiter" in self.config["div"]:
            limiter = self.config["div"]["limiter"]
        else:
            warnings.warn(
                "FDM: no limiter is specified. Use `none` as default."
            )
            limiter = "none"

        if var_j.name == var_i.name:
            # If var_i and var_j are the same field, use the same tensor.
            var_j.set_var_tensor(var_i().clone())

        div = []

        dx = var_j.dx

        for i in range(var_i.dim):

            d_val = torch.zeros_like(var_i()[i])

            for j in range(var_j.dim):

                if limiter == "none":
                    """Central difference scheme."""

                    pad = create_pad(var_i.mesh.dim)
                    slicer = inner_slicer(var_i.mesh.dim)

                    bc_il = var_i.get_bc(f"d-{NUM_TO_DIR[j]}l")
                    bc_ir = var_i.get_bc(f"d-{NUM_TO_DIR[j]}r")

                    # m_val = fill_pad(pad(var_i()[i]), j, 1, slicer)
                    m_val = fill_pad_bc(
                        pad(var_i()[i]), 1, slicer, [bc_il, bc_ir], j
                    )

                    d_val += (
                        var_j()[j]
                        * (
                            (
                                torch.roll(m_val, -1, j)
                                - torch.roll(m_val, 1, j)
                            )
                            / (2 * dx[j])
                        )[slicer]
                    )

                elif limiter == "upwind":

                    pad = create_pad(var_i.mesh.dim)
                    slicer = inner_slicer(var_i.mesh.dim)

                    var_i_pad = fill_pad(pad(var_i()[i]), j, 1, slicer)
                    var_j_pad = fill_pad(pad(var_j()[j]), j, 1, slicer)

                    m_val_p = (torch.roll(var_j_pad, -1, j) + var_j_pad) / 2
                    m_val_m = (torch.roll(var_j_pad, 1, j) + var_j_pad) / 2

                    f_val_p = (m_val_p + m_val_p.abs()) * var_i_pad / 2 - (
                        m_val_p - m_val_p.abs()
                    ) * torch.roll(var_i_pad, -1, j) / 2

                    f_val_m = (m_val_m + m_val_m.abs()) * torch.roll(
                        var_i_pad, 1, j
                    ) / 2 - (m_val_p - m_val_p.abs()) * var_i_pad / 2

                    d_val += ((f_val_p - f_val_m) / dx[j])[slicer]

                elif limiter == "quick":
                    pad = create_pad(var_i.mesh.dim, 2)
                    slicer = inner_slicer(var_i.mesh.dim, 2)

                    pass
                else:
                    raise ValueError("FDM: Unknown limiter.")

            div.append(d_val)

        return torch.stack(div)

    def grad(self, var: Field) -> Tensor:
        r"""Explicit discretization: Gradient

        Args:
            var: Field object to be discretized ($\Phi$).

        Returns:
            dict[int, dict[str, Tensor]]: returns jacobian of input Field. if `var.dim` is 1, it will be equivalent to `grad` of scalar field.
        """

        grad = []
        dx = var.dx

        pad = create_pad(var.mesh.dim)
        slicer = inner_slicer(var.mesh.dim)

        for i in range(var.dim):

            g_val = []

            for j in range(var.mesh.dim):

                bc_l = var.get_bc(f"d-{NUM_TO_DIR[j]}l")
                bc_r = var.get_bc(f"d-{NUM_TO_DIR[j]}r")

                var_padded = fill_pad_bc(
                    pad(var()[i]), 1, slicer, [bc_l, bc_r], j
                )

                g_val.append(
                    (
                        torch.roll(var_padded, -1, j)
                        - torch.roll(var_padded, 1, j)
                    )[slicer]
                    / (2 * dx[j])
                )
            grad.append(torch.stack(g_val))

        return torch.stack(grad)

    def laplacian(self, gamma: float, var: Field) -> Tensor:
        r"""Explicit discretization: Laplacian

        Args:
            var: Field object to be discretized ($\Phi$).

        Returns:
            dict[int, Tensor]: resulting `torch.Tensor`. `int` represents variable dimension, and `str` indicates `Mesh` dimension.

        """

        laplacian = []

        dx = var.dx

        for i in range(var.dim):

            l_val = torch.zeros_like(var()[i])

            for j in range(var.mesh.dim):
                ddx = dx[j] ** 2

                l_val_1d = (
                    torch.roll(var()[i], -1, j)
                    - 2 * var()[i]
                    + torch.roll(var()[i], 1, j)
                ) / ddx

                # Treat BC
                bc_l = var.get_bc(f"d-{NUM_TO_DIR[j]}l")
                bc_r = var.get_bc(f"d-{NUM_TO_DIR[j]}r")

                if bc_l is not None and bc_l.bc_treat:
                    # Mask forwards
                    mask_f = torch.roll(bc_l.bc_mask, 1, j)
                    mask_ff = torch.roll(bc_l.bc_mask, 2, j)

                    l_val_1d[mask_f] = (
                        -2 / 3 * var()[i][mask_f] + 2 / 3 * var()[i][mask_ff]
                    ) / ddx

                if bc_r is not None and bc_r.bc_treat:
                    # Mask backward
                    mask_b = torch.roll(bc_r.bc_mask, -1, j)
                    mask_bb = torch.roll(bc_r.bc_mask, -2, j)

                    l_val_1d[mask_b] = (
                        -2 / 3 * var()[i][mask_b] + 2 / 3 * var()[i][mask_bb]
                    ) / ddx

                l_val += l_val_1d * gamma

            laplacian.append(l_val)

        return torch.stack(laplacian)
