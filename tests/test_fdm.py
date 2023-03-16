#!/usr/bin/env python3
"""Test fvc (finite difference - volume centered) solver module.

Need to be revised. I comfused concept of fvc and fvm.
"""
import pytest
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.mesh.tools import inner_slicer
from pyapes.core.solver.fdc import FDC
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import homogeneous_bcs
from pyapes.testing.burgers import burger_exact_nd


def test_fdc_edge() -> None:
    """Test in 2D."""

    mesh = Mesh(Box[0:1, 0:1], None, [5, 5])

    # Field boundaries are all set to zero
    var = Field("test", 1, mesh, {"domain": None, "obstacle": None})

    # Arbitrary but useful initial condition
    var <<= 0.3 * mesh.X**2

    fdc = FDC()

    # Edge grad
    grad_torch = torch.gradient(var()[0], spacing=mesh.dx[0].item(), edge_order=2)
    grad_fdc = fdc.grad(var, edge=True)

    assert_close(grad_torch[0], grad_fdc[0][0])

    # Edge lap
    lap_torch_1 = torch.gradient(grad_torch[0], spacing=mesh.dx[0].item(), edge_order=2)
    lap_torch_2 = torch.gradient(grad_torch[1], spacing=mesh.dx[0].item(), edge_order=2)

    lap_torch = lap_torch_1[0] + lap_torch_2[0]

    lap_fdc = fdc.laplacian(var, edge=True)

    assert_close(lap_torch, lap_fdc[0])

    # Edge div
    pass


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [0.2]],
        [Box[0:1, 0:1], [0.2, 0.2]],
        [Box[0:1, 0:1, 0:1], [0.2, 0.2, 0.2]],
    ],
)
def test_fdc_ops(domain: Box, spacing: list[float]) -> None:
    """Test FDC module that discretizes current field values.
    Since the Neumann BC is a special case, we test with that. The dirichlet BC is straightforward and self explanatory, so we skip it.
    """

    mesh = Mesh(domain, None, spacing)
    slicer = inner_slicer(mesh.dim)

    # Since Neumann BC is set by \nabla Phi \cdot \vec{n} = V, below
    # BC will assign -2.0 to the lower boundary and 2.0 to the upper boundary.
    f_bc = homogeneous_bcs(mesh.dim, 2.0, "neumann")

    # Field boundaries are all set to zero
    var = Field("test", 1, mesh, {"domain": f_bc, "obstacle": None})

    # Arbitrary initial condition
    var <<= 0.3 * mesh.X**2

    # Check BC assignment
    for bc in var.bcs:
        bc.apply(var(), var.mesh.grid, 0)

    # Check Neumann BC
    # Lower BC
    phi0 = (-3 / 2 * var()[0][0] + 2 * var()[0][1] - 1 / 2 * var()[0][2]) / mesh.dx[0]
    # Upper BC
    phiN = (3 / 2 * var()[0][-1] - 2 * var()[0][-2] + 1 / 2 * var()[0][-3]) / mesh.dx[0]

    # Check Neumann BCs in the second order accuracy
    assert_close(phi0.mean(), torch.tensor(2.0))
    assert_close(phiN.mean(), torch.tensor(2.0))

    # Set discretizer
    fdc = FDC()

    # Laplacian operator
    lap = fdc.laplacian(var)
    lap_manuel = _lap_manuel_op(var()[0], mesh.dx, mesh.dim)

    assert_close(lap[0][slicer], lap_manuel[slicer])

    # Test reset function
    assert fdc.laplacian.A_coeffs is not None

    fdc.laplacian.reset()

    assert fdc.laplacian.A_coeffs is None
    assert fdc.laplacian.rhs_adj is None

    # Grad operator
    grad = fdc.grad(var)
    grad_manuel = _grad_manuel_op(var()[0], mesh.dx, mesh.dim)

    assert_close(grad[0][0][slicer], grad_manuel[0][slicer])

    # Div operator
    div = fdc.div(var)


def _grad_manuel_op(var: Tensor, dx: Tensor, dim: int) -> list[Tensor]:
    """Only test grad-x"""

    dx = dx[0]

    grad_manuel = []

    grad_manuel.append((torch.roll(var, -1, 0) - torch.roll(var, 1, 0)) / (2 * dx))

    x_inner = (torch.roll(var, -1, 0) - torch.roll(var, 1, 0)) / (2 * dx)
    # Var has no dependency on y direction
    # In y direction
    x_inner[1] = (4 / 3 * var[2] - 4 / 3 * var[1]) / (2 * dx)
    x_inner[-2] = (-4 / 3 * var[-2] + 4 / 3 * var[-3]) / (2 * dx)

    if dim == 1:
        grad_manuel[0] = x_inner
    elif dim == 2:
        # Need x_inner bc treats
        grad_manuel[0][:, 1] = x_inner[:, 1]
        grad_manuel[0][:, -2] = x_inner[:, -2]

        grad_manuel[0][1, :] = x_inner[1, :]
        grad_manuel[0][-2, :] = x_inner[-2, :]
    else:
        # Need x_in[0]ner bc treats
        grad_manuel[0][:, :, 1] = x_inner[:, :, 1]
        grad_manuel[0][:, :, -2] = x_inner[:, :, -2]

        grad_manuel[0][:, 1, :] = x_inner[:, 1, :]
        grad_manuel[0][:, -2, :] = x_inner[:, -2, :]

        grad_manuel[0][1, :, :] = x_inner[1, :, :]
        grad_manuel[0][-2, :, :] = x_inner[-2, :, :]

    return grad_manuel


def _lap_manuel_op(var: Tensor, dx: Tensor, dim: int) -> Tensor:
    dx = dx[0]

    lap_manuel = torch.zeros_like(var)
    for i in range(dim):
        lap_manuel += (
            torch.roll(var, -1, i) - 2 * var + torch.roll(var, 1, i)
        ) / dx**2

    x_inner = (torch.roll(var, -1, 0) - 2 * var + torch.roll(var, 1, 0)) / dx**2
    # Var has no dependency on y direction
    # In y direction
    x_inner[1] = (2 / 3 * var[2] - 2 / 3 * var[1]) / dx**2
    x_inner[-2] = (-2 / 3 * var[-2] + 2 / 3 * var[-3]) / dx**2

    if dim == 1:
        lap_manuel = x_inner
    elif dim == 2:
        # Need x_inner bc treats
        lap_manuel[:, 1] = x_inner[:, 1]
        lap_manuel[:, -2] = x_inner[:, -2]

        lap_manuel[1, :] = x_inner[1, :]
        lap_manuel[-2, :] = x_inner[-2, :]
    else:
        # Need x_inner bc treats
        lap_manuel[:, :, 1] = x_inner[:, :, 1]
        lap_manuel[:, :, -2] = x_inner[:, :, -2]

        lap_manuel[:, 1, :] = x_inner[:, 1, :]
        lap_manuel[:, -2, :] = x_inner[:, -2, :]

        lap_manuel[1, :, :] = x_inner[1, :, :]
        lap_manuel[-2, :, :] = x_inner[-2, :, :]

    return lap_manuel


@pytest.mark.parametrize(
    ["domain", "spacing"],
    [
        [Box[0:1], [0.2]],
        [Box[0:1, 0:1], [0.2, 0.2]],
        [Box[0:1, 0:1, 0:1], [0.2, 0.2, 0.2]],
    ],
)
def test_solver_fdm_ops(domain: Box, spacing: list[float]) -> None:
    """Test FDM module."""

    mesh = Mesh(domain, None, spacing)

    # Field boundaries are all set to zero
    var_i = Field("test_Fi", 1, mesh, None)
    var_j = Field("test_Fj", 1, mesh, None, init_val=5.0)

    var_i.set_var_tensor(2 * mesh.X**2)

    solver = Solver(None)
    fdm = FDM({"div": {"limiter": "upwind"}})

    # Poisson equation.
    solver.set_eq(fdm.laplacian(2.0, var_i) == 0.0)

    target = (
        (torch.roll(var_i()[0], -1, 0) - 2 * var_i()[0] + torch.roll(var_i()[0], 1, 0))
        / (mesh.dx[0] ** 2)
        * 2.0
    )

    assert_close(solver.Aop(var_i)[0][~mesh.t_mask], target[~mesh.t_mask])
    assert_close(torch.zeros_like(solver.Aop(var_i)), solver.rhs)

    var_i.set_var_tensor(4 * mesh.X**2)

    # Test call by reference
    assert_close(solver.Aop(var_i)[0][~mesh.t_mask], target[~mesh.t_mask] * 2)

    solver.set_eq(fdm.div(var_j, var_i) + fdm.laplacian(3.0, var_i) == 2.0)

    t_div = (var_i()[0] - torch.roll(var_i()[0], 1, 0)) / mesh.dx[0] * 5.0

    t_laplacian = (
        (torch.roll(var_i()[0], -1, 0) - 2 * var_i()[0] + torch.roll(var_i()[0], 1, 0))
        / (mesh.dx[0] ** 2)
        * 3.0
    )

    target = t_div[~mesh.t_mask] + t_laplacian[~mesh.t_mask]

    assert fdm.config["div"]["limiter"] == "upwind"
    assert_close(solver.Aop(var_i)[0][~mesh.t_mask], target)

    # Test 1D Advection Diffusion to see fdm.grad and fdm.div are interchangeable for this case.
    if mesh.dim == 1:
        solver.set_eq(fdm.grad(var_i) - fdm.laplacian(3.0, var_i) == 2.0)

        t_grad = (torch.roll(var_i()[0], -1, 0) - torch.roll(var_i()[0], 1, 0)) / (
            2 * mesh.dx[0]
        )

        t_laplacian = (
            (
                torch.roll(var_i()[0], -1, 0)
                - 2 * var_i()[0]
                + torch.roll(var_i()[0], 1, 0)
            )
            / (mesh.dx[0] ** 2)
            * 3.0
        )

        target = t_grad[~mesh.t_mask] - t_laplacian[~mesh.t_mask]
        assert_close(solver.Aop(var_i)[0][~mesh.t_mask], target)

    # Transient advection diffusion equation test
    dt = 0.01
    var_i.set_time(dt, 0.0)
    var_old = torch.rand_like(var_i())
    var_i.VARo = var_old
    rhs = torch.rand_like(var_i())

    solver.set_eq(
        fdm.ddt(var_i) + fdm.div(var_j, var_i) + fdm.laplacian(3.0, var_i) == rhs
    )
    t_div = (var_i()[0] - torch.roll(var_i()[0], 1, 0)) / mesh.dx[0] * 5.0

    t_laplacian = (
        (torch.roll(var_i()[0], -1, 0) - 2 * var_i()[0] + torch.roll(var_i()[0], 1, 0))
        / (mesh.dx[0] ** 2)
        * 3.0
    )

    d_t_var = (var_i()[0] - var_old[0]) / dt

    target = d_t_var + (t_div + t_laplacian)
    t_rhs = rhs

    assert fdm.config["div"]["limiter"] == "upwind"
    assert_close(solver.Aop(var_i)[0][~mesh.t_mask], target[~mesh.t_mask])
    if solver.rhs is not None:
        assert_close(solver.rhs, t_rhs)


class TestPrototype_LaplacianCoeffs:
    f_test = torch.linspace(0, 1, 6) ** 2

    def test_dirichlet(self):
        Am = torch.ones_like(self.f_test)
        Ap = torch.ones_like(Am)
        Ac = -2.0 * torch.ones_like(Am)

        lap_manuel = self.lap_op(self.f_test)
        lap_A_ops = self.lap_op(self.f_test, [Ap, Ac, Am])

        assert_close(lap_manuel, lap_A_ops)

    def test_neumann(self):
        """Symmetric is same as Neumann but only the zero gradient is applied."""

        # construct Neumann BC applied field
        lap_manuel = self.lap_op(self.f_test)
        lap_manuel[1] = -2 / 3 * self.f_test[1] + 2 / 3 * self.f_test[2]
        lap_manuel[-2] = -2 / 3 * self.f_test[-2] + 2 / 3 * self.f_test[-3]

        Am = torch.ones_like(self.f_test)
        Ap = torch.ones_like(Am)
        Ac = -2.0 * torch.ones_like(Am)

        # mask_forward
        Ac[1] = -2 / 3
        Ap[1] = 2 / 3
        Am[1] = 0

        # mask_backward
        Am[-2] = 2 / 3
        Ac[-2] = -2 / 3
        Ap[-2] = 0

        lap_A_ops = self.lap_op(self.f_test, [Ap, Ac, Am])

        assert_close(lap_manuel[1:-1], lap_A_ops[1:-1])

    def test_periodic(self):
        lap_manuel = self.lap_op(self.f_test)
        lap_manuel[1] = -2 * self.f_test[1] + self.f_test[2]
        lap_manuel[-2] = -2 * self.f_test[-2] + self.f_test[-3]

        Am = torch.ones_like(self.f_test)
        Ap = torch.ones_like(Am)
        Ac = -2.0 * torch.ones_like(Am)

        # mask_forward
        Am[1] = 0

        # mask_backward
        Ap[-2] = 0

        lap_A_ops = self.lap_op(self.f_test, [Ap, Ac, Am])

        assert_close(lap_manuel[1:-1], lap_A_ops[1:-1])

    def lap_op(self, field: Tensor, A_ops: list[Tensor] | None = None):
        """If A_ops is given, the list of Aop should be in order of `[Ap, Ac, Am]`."""

        if A_ops is None:
            return torch.roll(field, -1) - 2 * field + torch.roll(field, 1)
        else:
            return (
                A_ops[0] * torch.roll(field, -1)
                + A_ops[1] * field
                + A_ops[2] * torch.roll(field, 1)
            )
