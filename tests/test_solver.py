#!/usr/bin/env python3
from math import cos
from math import cosh
from math import exp
from math import pi
from math import sin

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close  # type: ignore

from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import homogeneous_bcs
from pyapes.core.variables.bcs import mixed_bcs
from pyapes.testing.burgers import burger_exact_nd
from pyapes.testing.poisson import poisson_bcs
from pyapes.testing.poisson import poisson_exact_nd
from pyapes.testing.poisson import poisson_rhs_nd

DISPLAY_PLOT: bool = True


def func_n1(
    grid: tuple[Tensor, ...], mask: Tensor, _, n_vec: Tensor
) -> Tensor:
    """Return the value of the Neumann boundary condition (sin(5x)).
    Boundary function should have 3 inputs and last one is variable itself.
    Since here, the variable is not used, it is set to dummy.
    """

    if n_vec.sum() > 0:
        return torch.sin(5.0 * grid[0][mask])
    else:
        return -torch.sin(5.0 * grid[0][mask])


@pytest.mark.parametrize(["dim"], [[1], [2], [3]])
def test_solver_tools(dim: int) -> None:
    """Testing `create_pad`, `inner_slicer` and `fill_pad` functions."""

    from pyapes.core.mesh.tools import create_pad, inner_slicer
    from pyapes.core.solver.tools import fill_pad

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
        [Box[0:1], [11], 1],
        [Box[0:1, 0:1], [0.01, 0.01], 2],
        [Box[0:1, 0:1, 0:1], [0.1, 0.1, 0.1], 3],
    ],
)
def test_poisson_nd_pure_dirichlet(
    domain: Box, spacing: list[float], dim: int
) -> None:
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
        {
            "fdm": {
                "method": "cg",
                "tol": 1e-6,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == rhs)
    solver.solve()

    assert solver.report["converge"] == True
    assert_close(var()[0], sol_ex, rtol=0.1, atol=0.01)

    var = var.zeros_like()

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-6,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    solver.set_eq(fdm.laplacian(1.0, var) == rhs)
    solver.solve()

    assert solver.report["converge"] == True
    assert_close(var()[0], sol_ex, rtol=0.1, atol=0.01)


def test_poisson_1d_periodic() -> None:
    """
    Solves the Poisson equation: d^2 V/dx^2 = sin(x), using second order centered
    finite differences, with periodic boundary conditions V(0) = V(L)
    The analytical solution is V(x) = -sin(x).

    Reference: http://boccelliengineering.altervista.org/junk/poisson_octave_fortran/simple_poisson_solvers.html
    """

    # Construct mesh
    mesh = Mesh(Box[0:pi], None, [64])

    # xl - xr - yl - yr
    f_bc = mixed_bcs(
        [None, None],
        ["periodic", "periodic"],
    )  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0)

    rhs = torch.zeros_like(var())
    rhs[0] = torch.sin(mesh.X)

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-8,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(var) == rhs)
    solver.solve()

    assert_close(var()[0], -torch.sin(mesh.X), rtol=0.1, atol=0.01)

    if DISPLAY_PLOT:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.plot(mesh.X, -torch.sin(mesh.X), "b-", label="Exact")
        ax.plot(mesh.X, var()[0], "r^", label="FDM")
        ax.legend(loc="lower right")
        plt.show()


def test_heat_conduction_2d_mixed() -> None:
    r"""Heat conduction (the Laplace equation) in 2D with mixed boundary conditions.

    Given is the heat conduction equation:

    .. math:
        \nabla^2 T = 0

    where, `T(xr) = T(yr) = 0`, `T'(xl) = 0`, and `T'(yl) = 1`.

    Reference: https://folk.ntnu.no/leifh/teaching/tkt4140/._main056.html
    """

    # Construct mesh
    mesh = Mesh(Box[0:1, 0:1], None, [11, 11])

    # xl - xr - yl - yr
    f_bc = mixed_bcs(
        [0.0, 0.0, 0.0, 1.0],
        ["neumann", "dirichlet", "neumann", "dirichlet"],
    )  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0)

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-8,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(var) == 0.0)
    solver.solve()

    def _exact_solution(x: Tensor, y: Tensor, n: int) -> Tensor:

        sol_ex = torch.zeros_like(x)

        for i in range(1, n + 1):
            lambda_n = (2 * i - 1) * pi / 2
            An = 2 * (-1) ** (i - 1) / (lambda_n * cosh(lambda_n))
            sol_ex += An * torch.cosh(lambda_n * y) * torch.cos(lambda_n * x)

        return sol_ex

    sol_ex = _exact_solution(mesh.X, mesh.Y, 200)

    import pandas as pd

    ref_data = torch.from_numpy(
        pd.read_csv(
            "./tests/data/laplace_equation/sol_ref_10_by_10.csv", index_col=0
        ).to_numpy()
    )

    assert_close(var()[0][:-1, :-1], ref_data, atol=0.01, rtol=0.01)

    if DISPLAY_PLOT:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
        axes[0].plot_surface(mesh.X, mesh.Y, var()[0], cmap="coolwarm")  # type: ignore
        axes[0].set_title("FDM")
        axes[1].plot_surface(mesh.X, mesh.Y, sol_ex, cmap="coolwarm")  # type: ignore
        axes[1].set_title("Exact")
        plt.show()


def test_poisson_2d_pure_periodic() -> None:
    """
    Reference: https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/periodic/python/documentation.html
    """

    # Construct mesh
    mesh = Mesh(Box[-1:1, -1:1], None, [128, 128])

    # xl - xr - yl - yr
    f_bc = mixed_bcs(
        [None, None, None, None],
        ["periodic", "periodic", "periodic", "periodic"],
    )  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0)
    rhs = torch.zeros_like(var())
    rhs[0] = torch.exp(-10.0 * (mesh.X**2 + mesh.Y**2))

    solver = Solver(
        {
            "fdm": {
                "method": "cg",
                "tol": 1e-6,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(var) == rhs)
    solver.solve()

    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 2)
    ax[0].contourf(mesh.X, mesh.Y, rhs[0], 20, cmap="coolwarm")
    ax[1].contourf(mesh.X, mesh.Y, var()[0], 20, cmap="coolwarm")
    plt.show()


def test_poisson_2d_mixed_periodic() -> None:
    """
    Reference: https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/periodic/python/documentation.html
    """

    # Construct mesh
    mesh = Mesh(Box[0:1, 0:1], None, [21, 21])

    # xl - xr - yl - yr
    f_bc = mixed_bcs(
        [None, None, 0, 0],
        ["periodic", "periodic", "dirichlet", "dirichlet"],
    )  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0)
    rhs = torch.zeros_like(var())
    rhs[0] = mesh.X * torch.sin(5.0 * pi * mesh.Y) + torch.exp(
        -((mesh.X - 0.5) ** 2 + (mesh.Y - 0.5) ** 2) / 0.02
    )

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-8,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(-fdm.laplacian(var) == rhs)
    solver.solve()

    if DISPLAY_PLOT:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(mesh.X, mesh.Y, var()[0], cmap="coolwarm")  # type: ignore
        plt.show()


def test_poisson_1d_mixed_neumann() -> None:
    """Referenced from https://www.scirp.org/pdf/JEMAA_2014092510103331.pdf

    The equation is given by

    math::
        d^2 phi / dx^2 = V0 cos(kx + phi0)

    which is underlined in the domain [-pi/2, pi/4].
    If we set V0 = 0, k = pi/2, and phi0 = pi/4 with the boundary conditions
    phi'(-pi/2) = 1/4 and phi(pi/4) = -1/2, the analytic solution is given by

    math::
        phi(x) = [1/4 - 2/pi sin(-pi/2 * pi/2 + pi/4)] * (x - pi/4)
                - 4/pi^2 [ cos(pi/2 x + pi/4) - cos(pi^2/8 + pi/4 )] - 1/2

    """
    # Construct mesh
    mesh = Mesh(Box[-pi / 2 : pi / 4], None, [101])

    f_bc = mixed_bcs([-1 / 4, -1 / 2], ["neumann", "dirichlet"])  # BC config
    # Target variable
    var = Field(
        "phi", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0
    )
    rhs = torch.zeros_like(var())
    rhs[0] = torch.cos(pi / 2 * mesh.X + pi / 4)

    sol_ex = torch.zeros_like(var())
    sol_ex[0] = (
        (1 / 4 - 2 / pi * sin(-(pi**2) / 4 + pi / 4)) * (mesh.X - pi / 4)
        - (4 / pi**2)
        * (torch.cos(pi / 2 * mesh.X + pi / 4) - cos(pi**2 / 8 + pi / 4))
        - 1 / 2
    )

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-6,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == rhs)
    solver.solve()

    # Check gradient at the boundary
    phi0 = (
        -3 / 2 * var()[0][0] + 2 * var()[0][1] - 1 / 2 * var()[0][2]
    ) / mesh.dx[0]
    phi0_ex = (
        -3 / 2 * sol_ex[0][0] + 2 * sol_ex[0][1] - 1 / 2 * sol_ex[0][2]
    ) / mesh.dx[0]

    assert_close(phi0, phi0_ex, atol=1e-3, rtol=1e-3)
    assert_close(var()[0], sol_ex[0], atol=1e-3, rtol=1e-3)


def test_poisson_2d_mixed_neumann() -> None:
    """Test the Poisson equation with BICGSTAB solver. (2D case)"""

    # Construct mesh
    mesh = Mesh(Box[0:0.5, 0:0.5], None, [101, 101])

    f_bc = mixed_bcs(
        [0, 0, 0, 0], ["dirichlet", "neumann", "dirichlet", "neumann"]
    )  # BC config

    # Target variable
    var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.0)
    rhs = torch.zeros_like(var())
    rhs[0] = -2 * pi**2 * torch.sin(pi * mesh.X) * torch.sin(pi * mesh.Y)

    solver = Solver(
        {
            "fdm": {
                "method": "cg",
                "tol": 1e-6,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    solver.set_eq(fdm.laplacian(1.0, var) == rhs)
    solver.solve()

    if DISPLAY_PLOT:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(mesh.X, mesh.Y, var()[0], cmap="coolwarm")  # type: ignore
        plt.show()


def test_advection_diffussion_1d() -> None:
    # Construct mesh
    mesh = Mesh(Box[0:1], None, [0.05])

    f_bc = homogeneous_bcs(1, 0.0, "dirichlet")

    # Target variable
    var = Field("U", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=0.5)

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-5,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM()

    epsilon = 0.5

    sol_ex = mesh.X - (
        torch.exp(-(1 - mesh.X) / epsilon) - exp(-1 / epsilon)
    ) / (1 - exp(-1 / epsilon))
    solver.set_eq(fdm.grad(var) - fdm.laplacian(epsilon, var) == 1.0)
    solver.solve()

    assert_close(var()[0], sol_ex, rtol=0.1, atol=0.01)


def test_burger_1d() -> None:
    # Construct mesh
    mesh = Mesh(Box[0 : 2 * pi], None, [101])

    solver = Solver(
        {
            "fdm": {
                "method": "bicgstab",
                "tol": 1e-5,
                "max_it": 1000,
                "report": True,
            }
        }
    )
    fdm = FDM({"div": {"limiter": "none"}})

    # Viscosity
    nu = 0.1
    sim_end = 0.1
    n_itr = 10
    dt = sim_end / n_itr

    res: list[Tensor] = []

    # Set dt to variable
    f_bc = homogeneous_bcs(1, None, "periodic")

    # Target variable
    init_val = burger_exact_nd(mesh, nu, 0.0)
    var = Field(
        "U", 1, mesh, {"domain": f_bc, "obstacle": None}, init_val=[init_val]
    )

    var.set_time(dt, 0.0)
    var.save_old()

    for _ in range(n_itr):

        res.append(var()[0].clone())

        solver.set_eq(
            fdm.ddt(var) + fdm.div(var, var) - fdm.laplacian(nu, var) == 0.0
        )
        solver.solve()
        var.update_time()

        sol_ex = burger_exact_nd(mesh, nu, var.t)

        assert_close(var()[0], sol_ex, rtol=0.01, atol=0.001)
