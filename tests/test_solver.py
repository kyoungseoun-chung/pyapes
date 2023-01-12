#!/usr/bin/env python3
from math import exp

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.solver.tools import create_pad
from pyapes.core.solver.tools import inner_slicer
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import homogeneous_bcs
from pyapes.testing.poisson import poisson_bcs
from pyapes.testing.poisson import poisson_exact_nd
from pyapes.testing.poisson import poisson_rhs_nd


@pytest.mark.parametrize(["dim"], [[1], [2], [3]])
def test_solver_tools(dim: int) -> None:
    """Testing `create_pad`, `inner_slicer` and `fill_pad` functions."""

    from pyapes.core.solver.tools import create_pad, inner_slicer, fill_pad

    var_entry = 3
    if dim == 1:
        var = torch.rand(var_entry)
    elif dim == 2:
        var = torch.rand(var_entry, var_entry)
    else:
        var = torch.rand(var_entry, var_entry, var_entry)

    pad_1 = create_pad(dim, 1)
    pad_2 = create_pad(dim, 2)

    slicer_1 = inner_slicer(dim, 1)
    slicer_2 = inner_slicer(dim, 2)

    var_padded_1 = fill_pad(pad_1(var), dim - 1, 1, slicer_1)
    var_padded_2 = fill_pad(pad_2(var), dim - 1, 2, slicer_2)

    if dim == 1:
        assert_close(var_padded_1[0], var_padded_1[slicer_1][0])
        assert_close(var_padded_1[-1], var_padded_1[slicer_1][-1])

        assert_close((var_padded_2[:2].sum() / 2), var_padded_2[slicer_2][0])
        assert_close((var_padded_2[-2:].sum() / 2), var_padded_2[slicer_2][-1])
    elif dim == 2:
        assert_close(var_padded_1[1:-1, 0], var_padded_1[slicer_1][:, 0])
        assert_close(var_padded_1[1:-1, -1], var_padded_1[slicer_1][:, -1])

        assert_close(
            (var_padded_2[1:-1, :2].sum(dim=1)[1:-1] / 2),
            var_padded_2[slicer_2][:, 0],
        )
        assert_close(
            (var_padded_2[1:-1, -2:].sum(dim=1)[1:-1] / 2),
            var_padded_2[slicer_2][:, -1],
        )
    else:
        assert_close(
            var_padded_1[1:-1, 1:-1, 0], var_padded_1[slicer_1][:, :, 0]
        )
        assert_close(
            var_padded_1[1:-1, 1:-1, -1], var_padded_1[slicer_1][:, :, -1]
        )

        assert_close(
            (var_padded_2[1:-1, 1:-1, :2].sum(dim=2)[1:-1, 1:-1] / 2),
            var_padded_2[slicer_2][:, :, 0],
        )
        assert_close(
            (var_padded_2[1:-1, 1:-1, -2:].sum(dim=2)[1:-1, 1:-1] / 2),
            var_padded_2[slicer_2][:, :, -1],
        )


@pytest.mark.parametrize(
    ["domain", "spacing", "dim"],
    [
        [Box[0:1], [0.01], 1],
        [Box[0:1, 0:1], [0.01, 0.01], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_poisson_nd(domain: Box, spacing: list[float], dim: int) -> None:
    """Test poisson in N-D cases.
    Note:
        - See `pyapes.testing.poisson` for more details.
    """

    # Construct mesh
    mesh = Mesh(domain, None, spacing)

    f_bc = poisson_bcs(dim)  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None})
    rhs = poisson_rhs_nd(mesh, var)  # RHS
    sol_ex = poisson_exact_nd(mesh)  # exact solution

    solver = Solver(
        {"fdm": {"method": "cg", "tol": 1e-6, "max_it": 1000, "report": True}}
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == fdm.rhs(rhs))
    solver.solve()

    assert_close(var()[0], sol_ex, rtol=0.1, atol=0.01)


def test_advection_diffussion_1d() -> None:
    # Construct mesh
    mesh = Mesh(Box[0:1], None, [0.05])

    f_bc = homogeneous_bcs(1, 0.0, "dirichlet")

    # Target variable
    var = Field("U", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.5)

    solver = Solver(
        {"fdm": {"method": "cg", "tol": 1e-5, "max_it": 1000, "report": True}}
    )
    fdm = FDM()

    epsilon = 0.1

    sol_ex = mesh.X - (
        torch.exp(-(1 - mesh.X) / epsilon) - exp(-1 / epsilon)
    ) / (1 - exp(-1 / epsilon))
    solver.set_eq(fdm.grad(var) - fdm.laplacian(epsilon, var) == 1.0)
    solver.solve()
    pass


def test_cg_AD_1d_simple() -> None:

    x = torch.linspace(0, 1, 101)
    dx = (x[1] - x[0]).item()
    var = torch.zeros_like(x)
    rhs = torch.ones_like(x)

    sol_ex = x - (torch.exp(-(1 - x) / 0.1) - exp(-1 / 0.1)) / (
        1 - exp(-1 / 0.1)
    )

    var_cg, tol_cg, itr_cg = cg_1d(var.clone(), rhs, dx)
    var_b, tol_b, itr_b = bicgstab_AD_1d(var.clone(), rhs, dx)

    pass


def Aop_AD_1d(var: Tensor, dx: float) -> Tensor:
    """Aop for 1D advection-diffusion equation."""

    adv = (torch.roll(var, -1, 0) - torch.roll(var, 1, 0)) / (2 * dx)
    diff = (torch.roll(var, -1, 0) - 2 * var + torch.roll(var, 1, 0)) / dx**2

    return adv - 0.1 * diff


def bicgstab_AD_1d(
    var: Tensor, rhs: Tensor, dx: float
) -> tuple[Tensor, float, int]:
    """Bi-conjugated gradient stabilized method."""

    # Padding for different dimensions
    pad = create_pad(1)

    tolerance = 1e-6
    max_it = 1000

    # Parameter initialization
    # Initial residue
    itr = 0

    var_new = var.clone()

    slicer = inner_slicer(1)

    # Initial residue
    r = pad(-rhs[slicer] + Aop_AD_1d(var_new, dx)[slicer])

    r0 = r.clone()
    v = torch.zeros_like(rhs)
    p = torch.zeros_like(v)
    t = torch.zeros_like(v)

    rho = 1.0
    alpha = 1.0
    omega = 1.0

    rho_next = torch.sum(r * r)
    tol = torch.sqrt(rho_next.max()).item()

    finished: bool = False

    while not finished:

        beta = rho_next / rho * alpha / omega
        rho = rho_next

        # Update p in-place
        p *= beta
        p -= beta * omega * v - r

        v = -pad(Aop_AD_1d(p, dx)[slicer])

        itr += 1

        alpha = torch.nan_to_num(
            rho / torch.sum(r0 * v), nan=0.0, posinf=0.0, neginf=0.0
        )
        s = r - alpha * v

        tol = torch.linalg.norm(s)

        if tol <= tolerance:
            var_new += alpha * p
            finished = True
            continue

        if itr >= max_it:
            break

        t = -pad(Aop_AD_1d(s, dx)[slicer])
        omega = torch.nan_to_num(
            torch.sum(t * s) / torch.sum(t * t),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rho_next = -omega * torch.sum(r0 * t)

        # Update residual
        r = s - omega * t

        # Update solution in-place-ish. Note that 'z *= omega' alters s if
        # precon = None. That's ok since s is no longer needed in this iter.
        # 'q *= alpha' would alter p.
        s *= omega
        var_new += s + alpha * p

        # BC
        var_new[0] = 0.0
        var_new[-1] = 0.0

        tol = torch.linalg.norm(r)

        if tol <= tolerance:
            finished = True

        if itr >= max_it:
            break

    return var_new, tol, itr


def cg_1d(var: Tensor, rhs: Tensor, dx: float) -> tuple[Tensor, float, int]:

    # Padding for different dimensions
    pad = create_pad(1)

    tolerance = 1e-6
    max_it = 1000

    # Parameter initialization
    # Initial residue
    tol = 1.0
    # Initial iterations
    itr = 0

    # Initial values
    Ad = torch.zeros_like(rhs)
    var_new = var.clone()

    slicer = inner_slicer(1)

    # Initial residue
    r = pad(-rhs[slicer] + Aop_AD_1d(var_new, dx)[slicer])

    d = r.clone()

    while tol > tolerance:

        var_old = var_new.clone()
        # CG steps
        # Aop in the search direction
        Ad = -pad(Aop_AD_1d(d, dx)[slicer])

        # Magnitude of the jump
        # Treat zero
        alpha = torch.nan_to_num(
            torch.sum(r * r) / torch.sum(d * Ad),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Iterated solution
        var_new = var_old + alpha * d

        # Intermediate computation
        beta_denom = torch.sum(r * r)

        # Update residual
        r -= alpha * Ad

        # Compute beta
        beta = torch.nan_to_num(
            torch.sum(r * r) / beta_denom, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Update search direction
        d = r + beta * d

        # BC
        var_new[0] = 0.0
        var_new[-1] = 0.0

        # Check validity of tolerance
        tol = torch.linalg.norm(var_new - var_old)

        # Add iteration counts
        itr += 1

        if itr > max_it:
            # Convergence is not achieved
            break

    return var_new, tol, itr
