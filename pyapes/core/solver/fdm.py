#!/usr/bin/env python3
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pyapes.core.geometry.basis import DIR
from pyapes.core.solver.tools import create_pad
from pyapes.core.solver.tools import fill_pad
from pyapes.core.solver.tools import inner_slicer
from pyapes.core.variables import Field


@dataclass
class FDM:
    """Collection of the operators for explicit finite difference discretizations."""

    config: Optional[dict[str, dict[str, str]]] = None

    def update_config(self, scheme: str, target: str, val: str):
        """Update config values."""

        if self.config is not None:
            self.config[scheme][target] = val
        else:
            warnings.warn("FDM: No config is specified.")

    def tensor(self, val: Tensor) -> Tensor:
        """Simply assign a given tensor and return. Mostly used for the RHS in `FVM`."""

        return val

    def div(self, var_i: Field, var_j: Field) -> dict[int, Tensor]:
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

        div: dict[int, Tensor] = {}

        dx = var_j.dx

        for i in range(var_i.dim):

            d_val = torch.zeros_like(var_i()[i])

            for j in range(var_j.dim):

                if limiter == "none":
                    """Central difference scheme."""

                    pad = create_pad(var_i.mesh.dim)
                    slicer = inner_slicer(var_i.mesh.dim)

                    m_val = fill_pad(
                        pad(var_i()[i] * var_j()[j]), j, 1, slicer
                    )

                    d_val += (
                        (torch.roll(m_val, -1, j) - torch.roll(m_val, 1, j))
                        / (2 * dx[j])
                    )[slicer]

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

            div.update({i: d_val})

        return div

    def grad(self, var: Field) -> dict[int, dict[str, Tensor]]:
        r"""Explicit discretization: Gradient

        Args:
            var: Field object to be discretized ($\Phi$).

        Returns:
            dict[int, dict[str, Tensor]]: returns jacobian of input Field. if `var.dim` is 1, it will be equivalent to `grad` of scalar field.
        """

        grad: dict[int, dict[str, Tensor]] = {}
        dx = var.dx

        pad = create_pad(var.mesh.dim)
        slicer = inner_slicer(var.mesh.dim)

        for i in range(var.dim):

            g_val: dict[str, Tensor] = {}

            for j in range(var.mesh.dim):

                var_padded = fill_pad(pad(var()[i]), j, 1, slicer)
                g_val.update(
                    {
                        DIR[j]: (
                            (
                                torch.roll(var_padded, -1, j)
                                - torch.roll(var_padded, 1, j)
                            )
                            / (2 * dx[j])
                        )[slicer]
                    }
                )
            grad[i] = g_val

        return grad

    def laplacian(self, gamma: float, var: Field) -> dict[int, Tensor]:
        r"""Explicit discretization: Laplacian

        Args:
            var: Field object to be discretized ($\Phi$).

        Returns:
            dict[int, Tensor]: resulting `torch.Tensor`. `int` represents variable dimension, and `str` indicates `Mesh` dimension.

        """

        laplacian: dict[int, Tensor] = {}

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

            laplacian.update({i: l_val})

        return laplacian


def _treat_edge(order: int):
    pass
