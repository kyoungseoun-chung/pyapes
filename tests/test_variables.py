#!/usr/bin/env python3
"""Test mesh"""
import pytest
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.geometry import Box
from pyapes.geometry.cylinder import Cylinder
from pyapes.mesh import Mesh
from pyapes.variables import Field
from pyapes.variables.bcs import BoxBoundary
from pyapes.variables.bcs import CylinderBoundary
from pyapes.variables.bcs import homogeneous_bcs


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [5]],
        [Box[0:1, 0:1], [5, 5]],
        [Box[0:1, 0:1, 0:1], [5, 5, 5]],
    ],
)
def test_field_bc_mask_individual(domain: Box, spacing: list[int]) -> None:
    mesh = Mesh(domain, None, spacing)

    f_bc = BoxBoundary(
        xl={"bc_type": "dirichlet", "bc_val": 0.0},
        xu={"bc_type": "dirichlet", "bc_val": 0.0},
        yl={"bc_type": "dirichlet", "bc_val": 0.0} if mesh.dim > 1 else None,
        yu={"bc_type": "dirichlet", "bc_val": 0.0} if mesh.dim > 1 else None,
        zl={"bc_type": "dirichlet", "bc_val": 0.0} if mesh.dim > 2 else None,
        zu={"bc_type": "dirichlet", "bc_val": 0.0} if mesh.dim > 2 else None,
    )

    var = Field("test", 1, mesh, {"domain": f_bc(), "obstacle": None})

    for i in range(2 * mesh.dim):
        target = var.bcs[i].bc_mask.clone()
        if i % 2 == 0:
            n_dir = -1
        else:
            n_dir = 1

        assert_close(var.bcs[i].bc_mask_prev, torch.roll(target, -n_dir, dims=i // 2))
        assert_close(
            var.bcs[i].bc_mask_prev2, torch.roll(target, -n_dir * 2, dims=i // 2)
        )
        assert_close(var.bcs[i].bc_mask_forward, torch.roll(target, n_dir, dims=i // 2))
        assert_close(
            var.bcs[i].bc_mask_forward2, torch.roll(target, n_dir * 2, dims=i // 2)
        )


def test_bc_config() -> None:
    f_bc = BoxBoundary(
        xl={"bc_type": "dirichlet", "bc_val": 0.44},
        xu={"bc_type": "neumann", "bc_val": 0},
        yl={"bc_type": "periodic", "bc_val": None},
        yu={"bc_type": "symmetry", "bc_val": None},
    )
    bc_config = [
        {"bc_face": "xl", "bc_type": "dirichlet", "bc_val": 0.44, "bc_val_opt": None},
        {"bc_face": "xu", "bc_type": "neumann", "bc_val": 0, "bc_val_opt": None},
        {"bc_face": "yl", "bc_type": "periodic", "bc_val": None, "bc_val_opt": None},
        {"bc_face": "yu", "bc_type": "symmetry", "bc_val": None, "bc_val_opt": None},
    ]

    assert f_bc() == bc_config

    f_bc = CylinderBoundary(
        rl={"bc_type": "dirichlet", "bc_val": 0.44},
        ru={"bc_type": "neumann", "bc_val": 0},
        zl={"bc_type": "periodic", "bc_val": None},
        zu={"bc_type": "symmetry", "bc_val": None},
    )
    bc_config = [
        {"bc_face": "rl", "bc_type": "dirichlet", "bc_val": 0.44, "bc_val_opt": None},
        {"bc_face": "ru", "bc_type": "neumann", "bc_val": 0, "bc_val_opt": None},
        {"bc_face": "zl", "bc_type": "periodic", "bc_val": None, "bc_val_opt": None},
        {"bc_face": "zu", "bc_type": "symmetry", "bc_val": None, "bc_val_opt": None},
    ]

    assert f_bc() == bc_config


@pytest.mark.parametrize(
    ["domain", "spacing", "dim"],
    [
        [Box[0:1], [0.1], 1],
        [Box[0:1, 0:1], [0.1, 0.1], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_fields(domain: Box, spacing: list[float], dim: int) -> None:
    """Test field boundaries."""

    mesh = Mesh(domain, None, spacing, "cpu", "double")

    var = Field("any", 1, mesh, {"domain": None, "obstacle": None})

    test_tensor = torch.rand(*var.size, dtype=mesh.dtype.float, device=mesh.device)

    var += test_tensor
    assert_close(var(), test_tensor)

    var /= var
    assert_close(var(), torch.ones_like(test_tensor))

    var *= 10
    assert_close(var(), torch.ones_like(test_tensor) * 10)

    var -= var
    assert_close(var(), test_tensor * 0)

    # Test Field copy
    var += 2.5
    copied_var = var.copy()
    assert_close(copied_var(), test_tensor * 0 + 2.5)

    zeroed_copied_var = var.zeros_like()
    assert_close(zeroed_copied_var(), test_tensor * 0)

    copied_var_name = var.copy(name="test_copy")
    assert copied_var_name.name == "test_copy"

    zeroed_copied_var_name = var.zeros_like(name="test_zeros_like")
    assert zeroed_copied_var_name.name == "test_zeros_like"


def test_cylinder_field_bcs() -> None:
    mesh = Mesh(Cylinder[0:1, 0:2], None, [5, 5])

    def ru_bc(grid: tuple[Tensor, ...], mask: Tensor, *_) -> Tensor:
        return grid[1][mask] * 4.4

    f_bc = CylinderBoundary(
        rl={"bc_type": "neumann", "bc_val": 0},
        ru={"bc_type": "dirichlet", "bc_val": ru_bc},
        zl={"bc_type": "neumann", "bc_val": 1.3},
        zu={"bc_type": "dirichlet", "bc_val": 0.44},
    )
    var = Field("d", 1, mesh, {"domain": f_bc(), "obstacle": None}, init_val="random")

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    rl_target = 4 / 3 * var()[0][1, 1:-1] - 1 / 3 * var()[0][2, 1:-1]
    zl_target = (
        4 / 3 * var()[0][1:-1, 1] - 1 / 3 * var()[0][1:-1, 2] + 2 / 3 * 1.3 * mesh.dx[1]
    )

    # Check ru
    assert_close(var()[0][-1, 1:-1], 4.4 * mesh.grid[1][0][1:-1])
    # Check zu
    assert_close(var()[0][1:-1, -1], 0.44 * torch.ones_like(var()[0][1:-1, -1]))
    # Check rl
    assert_close(var()[0][0, 1:-1], rl_target)
    # Check zl
    assert_close(var()[0][1:-1, 0], zl_target)

    def zu_bc(
        grid: tuple[Tensor, ...], mask: Tensor, _, opt: dict[str, Tensor]
    ) -> Tensor:
        val = torch.sum(opt["T"])

        return grid[0][mask] * val

    # Test bc_val_opt
    f_bc = CylinderBoundary(
        rl={"bc_type": "neumann", "bc_val": 0},
        ru={"bc_type": "dirichlet", "bc_val": ru_bc},
        zl={"bc_type": "neumann", "bc_val": 1.3},
        zu={
            "bc_type": "dirichlet",
            "bc_val": zu_bc,
            "bc_val_opt": {"T": torch.ones_like(var()[0])},
        },
    )
    var = Field("d", 1, mesh, {"domain": f_bc(), "obstacle": None}, init_val="random")

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    val = var()[0].numel()

    assert_close(var()[0][1:-1, -1], mesh.grid[0][1:-1, -1] * val)


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [0.1]],
        [Box[0:1, 0:1], [0.1, 0.1]],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1]],
    ],
)
def test_box_field_bcs(domain: Box, spacing: list[float]) -> None:
    """Test field boundaries."""

    mesh = Mesh(domain, None, spacing, "cpu", "double")

    f_bc_d = homogeneous_bcs(mesh.dim, 0.44, "dirichlet")
    var = Field("d", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random")

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    assert_close(var()[0][0].mean().item(), 0.44)
    assert_close(var()[0][-1].mean().item(), 0.44)

    f_bc_d = homogeneous_bcs(mesh.dim, 1.0, "neumann")
    var = Field("n", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random")

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    target_l = 4 / 3 * var()[0][1] - 1 / 3 * var()[0][2] - 2 / 3 * 1.0 * 0.1
    target_u = 4 / 3 * var()[0][-2] - 1 / 3 * var()[0][-3] + 2 / 3 * 1.0 * 0.1

    assert_close(var()[0][0], target_l)
    assert_close(var()[0][-1], target_u)

    f_bc_d = homogeneous_bcs(mesh.dim, None, "periodic")
    var = Field("p", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random")

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    if mesh.dim == 1:
        assert_close(var()[0][0], var()[0][-1])
        assert_close(var()[0][-1], var()[0][0])
    elif mesh.dim == 2:
        assert_close(var()[0][0, :], var()[0][-1, :])
        assert_close(var()[0][-1, :], var()[0][0, :])
    else:
        assert_close(var()[0][0, :, :], var()[0][-1, :, :])
        assert_close(var()[0][-1, :, :], var()[0][0, :, :])

    f_bc_d = homogeneous_bcs(mesh.dim, None, "symmetry")
    var = Field("s", 1, mesh, {"domain": f_bc_d, "obstacle": None}, init_val="random")

    bc_xl = var.get_bc("d-xl")

    if bc_xl is not None:
        assert bc_xl.type == "symmetry"
        assert bc_xl.bc_id == "d-xl"

    for bc in var.bcs:
        bc.apply(var(), mesh.grid, 0)

    if mesh.dim == 1:
        assert_close(var()[0][0], var()[0][1])
        assert_close(var()[0][-1], var()[0][-2])
    elif mesh.dim == 2:
        assert_close(var()[0][0, :], var()[0][1, :])
        assert_close(var()[0][-1, :], var()[0][-2, :])
    else:
        assert_close(var()[0][0, :, :], var()[0][1, :, :])
        assert_close(var()[0][-1, :, :], var()[0][-2, :, :])
