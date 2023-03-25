#!/usr/bin/env python3
"""Test roll vs conv1d discretization"""
import time

import torch
from torch.nn.functional import conv1d
from torch.testing import assert_close  # type: ignore

from pyapes.geometry import Cylinder
from pyapes.mesh import Mesh
from pyapes.variables import Field

torch.set_default_dtype(torch.float64)


def test_div_diff_flux() -> None:
    """Test for div(D * grad(var))"""

    from pyapes.solver.fdc import FDC
    from pyapes.tools.spatial import DiffFlux, ScalarOP

    mesh = Mesh(Cylinder[0:1, 0:1], None, [3, 3])
    var = Field("test", 1, mesh, {"domain": None, "obstacle": None})
    var.set_var_tensor(mesh.grid[0] ** 2)

    hess = ScalarOP.hess(var)

    FDC.div(hess, FDC.grad(var))

    div = FDC.div(DiffFlux(hess, var))


def test_ops_comp_grad():
    nx = int(2**12)

    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, nx)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = torch.meshgrid(x, y, indexing="ij")

    test_field = X**2 + 2 * Y**2

    grad_roll_x = (torch.roll(test_field, -1, 0) - torch.roll(test_field, 1, 0)) / (
        2 * dx
    )
    tic = time.perf_counter()
    grad_roll_y = (torch.roll(test_field, -1, 1) - torch.roll(test_field, 1, 1)) / (
        2 * dy
    )
    t_roll = time.perf_counter() - tic

    assert_close(grad_roll_x[1:-1, 1:-1], 2 * X[1:-1, 1:-1])
    assert_close(grad_roll_y[1:-1, 1:-1], 4 * Y[1:-1, 1:-1])

    k_cw = 4

    # Shape = (out_channels, in_channels, kernel_size)
    test_kernel = torch.zeros((k_cw, k_cw, 3))
    test_kernel[0, -1, 0] = -1
    test_kernel[-1, 0, -1] = 1
    diag = torch.ones(k_cw - 1)
    test_kernel[:, :, 1] = torch.diag(diag, 1) - torch.diag(diag, -1)

    test_kernel /= 2 * dy

    # Shape = (batch, in_channels, width)
    tic = time.perf_counter()
    test_field_T = torch.transpose(test_field.view(nx, -1, k_cw), 1, 2)
    conv_op = conv1d(test_field_T, test_kernel, stride=1, padding="same")
    grad_y = torch.reshape(torch.transpose(conv_op, 1, 2), (nx, nx))
    t_conv_y = time.perf_counter() - tic

    assert_close(grad_y[1:-1, 1:-1], 4 * Y[1:-1, 1:-1])

    tic = time.perf_counter()
    test_field_T = torch.transpose(test_field.T.view(nx, -1, k_cw), 1, 2)
    conv_op = conv1d(test_field_T, test_kernel, stride=1, padding="same")
    grad_x = torch.reshape(torch.transpose(conv_op, 1, 2), (nx, nx)).T
    t_conv_x = time.perf_counter() - tic

    assert_close(grad_x[1:-1, 1:-1], 2 * X[1:-1, 1:-1])
