#!/usr/bin/env python3
"""Collection of useful tools to be used in `pyABC.core.solver` module."""
from typing import Union

import torch
from torch import Tensor
from torch.nn import ConstantPad1d
from torch.nn import ConstantPad2d
from torch.nn import ConstantPad3d


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


def inner_slicer(dim: int, pad: int = 1) -> tuple:
    """Create tensor innder slicer."""

    if dim == 1:
        return (slice(pad, -pad),)
    elif dim == 2:
        return (slice(pad, -pad), slice(pad, -pad))
    else:
        return (slice(pad, -pad), slice(pad, -pad), slice(pad, -pad))
