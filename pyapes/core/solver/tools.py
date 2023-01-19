#!/usr/bin/env python3
"""Collection of useful tools to be used in `pyABC.core.solver` module."""
import torch
from torch import Tensor

from pyapes.core.variables.bcs import BC_type


def fill_pad_bc(
    var_padded: Tensor,
    pad_length: int,
    slicer: list[slice],
    bcs: list[BC_type | None],
    dim: int,
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

        var_padded = _bc_interpolate(var_padded, bcs, slicer, dim)
        var_padded[slicer] = var_inner

    return var_padded


def _bc_interpolate(
    var_padded: Tensor,
    bcs: list[BC_type | None],
    slicer: list[slice],
    dim: int,
) -> Tensor:
    """Interpolate boundary conditions to padded tensor."""

    var_interp = torch.zeros_like(var_padded)

    for idx, bc in enumerate(bcs):

        if bc is None:
            var_interp += torch.roll(var_padded, -1 + 2 * idx, dim)
        else:
            if bc.type == "dirichlet":
                var_interp += torch.roll(
                    var_padded, -bc.bc_n_dir, bc.bc_face_dim
                )
            elif bc.type == "neumann":
                var_dummy = torch.zeros_like(var_padded)

                var_rolled = torch.roll(
                    var_padded, bc.bc_n_dir, bc.bc_face_dim
                )
                del_var = var_rolled - var_padded

                var_dummy[slicer] = (var_padded - del_var)[slicer]
                var_interp += torch.roll(
                    var_dummy, bc.bc_n_dir, bc.bc_face_dim
                )
            elif bc.type == "symmetry":

                var_dummy = torch.zeros_like(var_padded)
                var_dummy[slicer] = var_padded[slicer]

                var_interp += torch.roll(
                    var_dummy, bc.bc_n_dir, bc.bc_face_dim
                )
            elif bc.type == "periodic":
                var_dummy = torch.zeros_like(var_padded)
                var_dummy[slicer] = var_padded[slicer]
                var_rolled = torch.roll(
                    var_dummy, bc.bc_n_dir * 2, bc.bc_face_dim
                )

                var_interp += torch.roll(
                    var_padded, bc.bc_n_dir * 2, bc.bc_face_dim
                )
            else:
                ValueError(f"BC: {bc.type} is not supported!")

    return var_interp


def fill_pad(
    var_padded: Tensor, dim: int, pad_length: int, slicer: list
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
