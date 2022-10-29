#!/usr/bin/env python3
"""Discretization of explicit field.
Results of discretization always returns torch.Tensor.
"""
from typing import Union
import torch
from torch import Tensor

from pyapes.core.variables import Field
from pyapes.core.geometry.basis import DIR
from pyapes.core.variables import Flux


class Source:
    """Create source/sink at the cell center."""

    def __new__(cls, *args, **kwargs):
        return "__call__"

    @staticmethod
    def __call__(val: Union[float, Tensor], var: Field) -> Flux:

        source = Flux(var.mesh)
        for i in range(var.dim):
            for j in range(var.mesh.dim):
                if 
                source.to_center(i, j, )


class Grad:
    r"""Explicit discretization: Gradient

    Args:
        var: Field object to be discretized ($\Phi$).

    Returns:
        dict[int, dict[str, Tensor]]: resulting `torch.Tensor`. `int` represents variable dimension, and `str` indicates `Mesh` dimension.

    """

    def __new__(cls, *args, **kwargs):
        return "__call__"

    @staticmethod
    def __call__(var: Field) -> Flux:

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
                    "l",
                    (var()[i] + torch.roll(var()[i], -1, j)) / 2,
                )

        # How to BCs?
        for bc, mask in zip(var.bcs, var.mesh.d_mask):
            bc.apply(mask, var(), grad, var.mesh.grid, 0)

        grad.sum_flux()

        return grad
