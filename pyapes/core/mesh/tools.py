#!/usr/bin/env python3
from typing import Union

from torch.nn import ConstantPad1d
from torch.nn import ConstantPad2d
from torch.nn import ConstantPad3d

from pyapes.core.geometry.basis import DIR_TO_NUM
from pyapes.core.geometry.basis import SIDE_TO_NUM
from pyapes.core.variables.bcs import BC_type


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


def boundary_slicer(dim: int, bcs: list[BC_type]) -> list[slice]:
    slice_idx: list[list[int | None]] = [[1, -1] for _ in range(dim)]

    for bc in bcs:
        if bc.bc_type == "periodic":
            d_idx = DIR_TO_NUM[bc.bc_face[0]]
            s_idx = SIDE_TO_NUM[bc.bc_face[1]]
            slice_idx[d_idx][s_idx] = None
        else:
            # Do nothing
            pass

    slicer = [slice(*slice_idx[i]) for i in range(dim)]
    return slicer


def inner_slicer(dim: int, pad: int | None = 1) -> list[slice]:
    """Create tensor inner slicer.

    Example:
        >>> inner_slicer(2) # create slicer for 2D mesh
        [slice(1, -1), slice(1, -1)]
        >>> inner_slicer(2, pad=2) # create slicer for 2D mesh with 2 pad
        [slicer(2, -2), slice(2, -2)]
    """

    return [slice(pad, -pad if isinstance(pad, int) else None) for _ in range(dim)]
