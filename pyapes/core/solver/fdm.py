#!/usr/bin/env python3
from dataclasses import dataclass

import torch
from torch import Tensor

from pyapes.core.geometry.basis import DIR
from pyapes.core.solver.tools import create_pad
from pyapes.core.solver.tools import inner_slicer
from pyapes.core.variables import Field


# NOTE: return Tensor? or dict?
# NOTE: BC with slicer?
@dataclass
class FDM:
    """Collection of the operators for explicit finite difference discretizations."""

    param: dict[str, dict[str, str]]

    def tensor(self, val: Tensor) -> Tensor:
        """Simply assign a given tensor and return. Mostly used for the RHS in `FVM`."""

        return val

    # WIP: Need a good idea for the boundary conditions
    def div(self, var_i: Field, var_j: Field) -> Tensor:
        """Divergence of two fields.
        Note:
            - To avoid the checkerboard problem, flux limiter is used. It supports `none`, `upwind` and `quick` limiter. (here, `none` is equivalent to the second order central difference.)
        """

        limitter = self.param["div"]["limitter"]

        dx = var_j.dx

        for i in range(var_i.dim):

            for j in range(var_i.mesh.dim):
                val = var_i()[i] * var_j()[j]
                div.to_face(
                    i,
                    DIR[j],
                    "l",
                    (val + torch.roll(val, 1, j)) / (0.5 * dx[j]),
                )
                div.to_face(
                    i,
                    DIR[j],
                    "r",
                    (torch.roll(val, -1, j) + val) / (0.5 * dx[j]),
                )

        return div.tensor()

    def grad(self, var: Field) -> dict[int, dict[str, Tensor]]:
        r"""Explicit discretization: Gradient

        Args:
            var: Field object to be discretized ($\Phi$).

        Returns:
            dict[int, dict[str, Tensor]]: returns jacobian of input Field. if `var.dim` is 1, it will be equivalent to `grad` of scalar field.
        """

        grad: dict[int, dict[str, Tensor]] = {}
        dx = var.dx

        for i in range(var.dim):
            for j in range(var.mesh.dim):

                grad.update(
                    {
                        i: {
                            DIR[j]: (
                                torch.roll(var()[i], -1, j)
                                - torch.roll(var()[i], 1, j)
                            )
                            / (2 * dx[j])
                        }
                    }
                )

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

        for i in range(var.dim):

            l_val = torch.zeros_like(var()[i])

            for j in range(var.mesh.dim):
                ddx = dx[j] ** 2
                l_val -= (
                    (
                        torch.roll(var()[i], -1, j)
                        - 2 * var()[i]
                        + torch.roll(var()[i], 1, j)
                    )
                    / ddx
                    * gamma
                )

            laplacian.update({i: l_val})

        return laplacian
