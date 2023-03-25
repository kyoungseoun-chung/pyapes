#!/usr/bin/env python3
from pyapes.geometry.basis import DIR_TO_NUM
from pyapes.geometry.basis import SIDE_TO_NUM
from pyapes.variables.bcs import BC_type


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
