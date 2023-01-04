#!/usr/bin/env python3
"""Collection of useful tools to be used in `pyABC.core.solver` module."""
from typing import Union

from torch.nn import ConstantPad1d
from torch.nn import ConstantPad2d
from torch.nn import ConstantPad3d


def create_pad(
    dim: int,
) -> Union[ConstantPad1d, ConstantPad2d, ConstantPad3d]:
    """Create pad object."""

    if dim == 1:
        return ConstantPad1d(1, 0)
    elif dim == 2:
        return ConstantPad2d(1, 0)
    else:
        return ConstantPad3d(1, 0)


def inner_slicer(dim: int) -> tuple:

    if dim == 1:
        return (slice(1, -1),)
    elif dim == 2:
        return (slice(1, -1), slice(1, -1))
    else:
        return (slice(1, -1), slice(1, -1), slice(1, -1))
