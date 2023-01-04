#!/usr/bin/env python3
"""Collection of useful tools to be used in `pyABC.core.solver` module."""
from typing import Union

from torch.nn import ConstantPad1d
from torch.nn import ConstantPad2d
from torch.nn import ConstantPad3d


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
