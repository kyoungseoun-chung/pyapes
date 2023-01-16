#!/usr/bin/env python3
"""Collection of useful tools to be used in `pyABC.core.solver` module."""
from typing import Union

import torch
from torch import Tensor
from torch.nn import ConstantPad1d
from torch.nn import ConstantPad2d
from torch.nn import ConstantPad3d

from pyapes.core.variables.bcs import BC_type

# WIP: experimental!
def fill_pad_bc(
    var_padded: Tensor,
    pad_length: int,
    slicer: tuple[slice, ...],
    bcs: list[BC_type],
) -> Tensor:
    """Interpolate boundary conditions to padded tensor.

    Args:
        var_padded (Tensor): padded tensor.
        pad_length (int): length of padding in one side.
        slicer (tuple): inner part slicer
        bcs (list[BC_type]): list of boundary conditions (in order of `l` to `r`).
    """

    # Make sure bcs are always for both sides l and r
    assert len(bcs) == 2

    var_inner = var_padded[slicer].clone()

    for _ in range(pad_length):

        var_padded = _bc_interpolate(var_padded, bcs, slicer)
        var_padded[slicer] = var_inner

    return var_padded


def _bc_interpolate(
    var_padded: Tensor, bcs: list[BC_type], slicer: tuple[slice, ...]
) -> Tensor:
    """Interpolate boundary conditions to padded tensor."""

    var_interp = torch.zeros_like(var_padded)

    for bc in bcs:

        if bc.type == "dirichlet":
            var_interp += torch.roll(var_padded, -bc.bc_n_dir, bc.bc_face_dim)
        elif bc.type == "neumann":
            var_dummy = torch.zeros_like(var_padded)

            var_rolled = torch.roll(var_padded, bc.bc_n_dir, bc.bc_face_dim)
            del_var = var_rolled - var_padded

            var_dummy[slicer] = (var_padded - del_var)[slicer]
            var_interp += torch.roll(var_dummy, bc.bc_n_dir, bc.bc_face_dim)
        elif bc.type == "symmetry":

            var_dummy = torch.zeros_like(var_padded)
            var_dummy[slicer] = var_padded[slicer]

            var_interp += torch.roll(var_dummy, bc.bc_n_dir, bc.bc_face_dim)
        elif bc.type == "periodic":
            var_interp += torch.roll(var_padded, bc.bc_n_dir, bc.bc_face_dim)
        else:
            ValueError(f"BC: {bc.type} is not supported!")

    return var_interp


def fill_pad(
    var_padded: Tensor, dim: int, pad_length: int, slicer: tuple
) -> Tensor:
    """Fill padded tensor with values from inner part.

    Note:
        - It copies the edge of the inner part and fill the padded part with the copied values.
        - Repeat process over `pad_length` times.

    Args:
        var_padded (Tensor): padded tensor. Filled with zeros (usually).
        dim (int): dimension to apply fill padding.
        pad_length (int): length of padding in one side.
        slicer (tuple): inner part slicer

    Returns:
        Tensor: padded tensor with values from inner part.
    """

    var_inner = var_padded[slicer].clone()
    for _ in range(pad_length):
        var_roll_p = torch.roll(var_padded, -1, dim)
        var_roll_m = torch.roll(var_padded, 1, dim)
        var_padded = var_roll_p + var_roll_m
        var_padded[slicer] = var_inner

    return var_padded


def create_pad(
    dim: int, pad_length: int = 1, pad_value: float = 0.0
) -> Union[ConstantPad1d, ConstantPad2d, ConstantPad3d]:
    """Create pad object."""

    if dim == 1:
        return ConstantPad1d(pad_length, pad_value)
    elif dim == 2:
        return ConstantPad2d(pad_length, pad_value)
    else:
        return ConstantPad3d(pad_length, pad_value)


def inner_slicer(dim: int, pad: int = 1) -> tuple[slice, ...]:
    """Create tensor innder slicer."""

    if dim == 1:
        return (slice(pad, -pad),)
    elif dim == 2:
        return (slice(pad, -pad), slice(pad, -pad))
    else:
        return (slice(pad, -pad), slice(pad, -pad), slice(pad, -pad))
