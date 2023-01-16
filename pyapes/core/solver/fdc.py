#!/usr/bin/env python3
"""Finite Difference for the current field `FDC`. Similar to Openfoam's `FVC` class."""
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pyapes.core.geometry.basis import NUM_TO_DIR
from pyapes.core.solver.tools import create_pad
from pyapes.core.solver.tools import fill_pad
from pyapes.core.solver.tools import fill_pad_bc
from pyapes.core.solver.tools import inner_slicer
from pyapes.core.variables import Field


@dataclass
class FDC:
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

        div = []

        dx = var_j.dx

        for i in range(var_i.dim):

            d_val = torch.zeros_like(var_i()[i])

            for j in range(var_j.dim):

                if limiter == "none":
                    """Central difference scheme."""

                    pad = create_pad(var_i.mesh.dim)
                    slicer = inner_slicer(var_i.mesh.dim)

                    m_val = fill_pad(pad(var_i()[i]), j, 1, slicer)

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
                    pad(var()[i]), 1, slicer, [bc_l, bc_r]
                )

                var_padded = fill_pad(pad(var()[i]), j, 1, slicer)
                g_val.append(
                    (
                        (
                            torch.roll(var_padded, -1, j)
                            - torch.roll(var_padded, 1, j)
                        )
                        / (2 * dx[j])
                    )[slicer]
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
        pad = create_pad(var.mesh.dim)
        slicer = inner_slicer(var.mesh.dim)

        for i in range(var.dim):

            l_val = torch.zeros_like(var()[i])

            for j in range(var.mesh.dim):
                ddx = dx[j] ** 2
                var_padded = fill_pad(pad(var()[i]), j, 1, slicer)

                l_val += (
                    (
                        torch.roll(var_padded, -1, j)
                        - 2 * var_padded
                        + torch.roll(var_padded, 1, j)
                    )
                    / ddx
                    * gamma
                )[slicer]

            laplacian.append(l_val)

        return torch.stack(laplacian)
