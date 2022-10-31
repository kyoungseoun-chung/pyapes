#!/usr/bin/env python3
"""Discretization of explicit field.
Results of discretization always returns torch.Tensor.
"""
from typing import Union

import torch
from torch import Tensor

from .fv import Discretizer
from pyapes.core.geometry.basis import DIR
from pyapes.core.variables import Field
from pyapes.core.variables import Flux


class FTensor(Discretizer):
    """Simple interface around `torch.Tensor`."""

    def __call__(self, val: Tensor) -> Tensor:

        return val


class Source(Discretizer):
    """Create source/sink at the cell center."""

    def __call__(self, val: Union[float, Tensor], var: Field) -> Tensor:

        source = Flux(var.mesh)

        for i in range(var.dim):
            for j in range(var.mesh.dim):
                source.to_center(
                    i,
                    DIR[j],
                    val
                    if isinstance(val, Tensor)
                    else val * torch.ones_like(var()),
                )
        return source.tensor()


class Grad(Discretizer):
    r"""Explicit discretization: Gradient

    Args:
        var: Field object to be discretized ($\Phi$).

    Returns:
        dict[int, dict[str, Tensor]]: resulting `torch.Tensor`. `int` represents variable dimension, and `str` indicates `Mesh` dimension.

    """

    def __call__(self, var: Field) -> Tensor:

        grad = Flux(var.mesh)

        for i in range(var.dim):
            for j in range(var.mesh.dim):

                grad.to_face(
                    i,
                    DIR[j],
                    "l",
                    (var()[i] + torch.roll(var()[i], 1, j)) / 2,
                )
                grad.to_face(
                    i,
                    DIR[j],
                    "r",
                    (var()[i] + torch.roll(var()[i], -1, j)) / 2,
                )

        for bc in var.bcs:
            bc.apply(var(), grad, var.mesh.grid, 0)

        grad.sum()

        return grad.tensor()


class Laplacian(Discretizer):
    r"""Explicit discretization: Laplacian

    Args:
        var: Field object to be discretized ($\Phi$).

    Returns:
        dict[int, dict[str, Tensor]]: resulting `torch.Tensor`. `int` represents variable dimension, and `str` indicates `Mesh` dimension.

    """

    def __call__(self, gamma: float, var: Field) -> Tensor:

        laplacian = Flux(var.mesh)

        dx = var.dx

        for i in range(var.dim):
            for j in range(var.mesh.dim):

                laplacian.to_face(
                    i,
                    DIR[j],
                    "l",
                    (var()[i] - torch.roll(var()[i], 1, j)) / dx[j],
                )
                laplacian.to_face(
                    i,
                    DIR[j],
                    "r",
                    (torch.roll(var()[i], -1, j) - var()[i]) / dx[j],
                )

        for bc in var.bcs:
            bc.apply(var(), laplacian, var.mesh.grid, 1)

        laplacian.sum_all()
        laplacian *= gamma

        return laplacian.tensor()


FVC_type = Union[Source, Grad, Laplacian, FTensor]
