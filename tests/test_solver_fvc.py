#!/usr/bin/env python3
"""Test solver components."""
import numpy as np
import torch

from pyABC.core.boundaries import BC_OBJECT_FACTORY
from pyABC.core.boundaries import get_basic_domain
from pyABC.core.fields import Variables
from pyABC.core.mesh import Mesh
from pyABC.solver.fdm import fdm_poisson
from pyABC.solver.operators import Solver
from pyABC.tools.utils import Container
from tests.test_variables import grid_setup

# Global settings
LX_MIN = [-0.05, -0.05, -0.05]
LX_MAX = [1.05, 1.05, 1.05]
LX, DX, NX, X = grid_setup(l_min=LX_MIN, l_max=LX_MAX, dx=[0.1, 0.1, 0.1])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mesh_for_solver_test() -> Mesh:

    mesh_config = Container(
        NAME="test_solver",
        LX_min=LX_MIN,
        LX_max=LX_MAX,
        DX=DX,
        OBJECTS=None,
        VARIABLES=None,
        REFERENCE=None,
        DEVICE="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create simple version of Mesh object
    MESH = Mesh(mesh_config, simple=True)

    return MESH


def scalar_field_for_solver_test(bc_types: list, bc_vals: list) -> Variables:

    obj_config, bcs_config = get_basic_domain(LX, bc_types, bc_vals)

    # Create Patch object
    objs = []
    for i, obj in enumerate(obj_config):
        objs.append(BC_OBJECT_FACTORY[obj["type"]](obj, i))

    # Create VAR
    VAR = Variables(
        name="P",
        type="Scalar",
        dim=1,
        objs=objs,
        bc_config=bcs_config,
        x=X,
        NX=NX,
        DX=DX,
        device=DEVICE,
    )

    return VAR


def test_poisson_2() -> None:
    """Test the Poisson solver with different BCs."""
    bc_types = [
        "dirichlet",
        "neumann",
        "symmetry",
        "periodic",
        "dirichlet",
        "dirichlet",
    ]
    bc_vals = [[0.0], [2.0], [0.0], [0.0], [1.0], [-1.0]]

    var = scalar_field_for_solver_test(bc_types, bc_vals)
    mesh = mesh_for_solver_test()

    # RHS of the Poisson equation
    rhs = torch.from_numpy(
        np.sin(np.pi * mesh.X1)
        * np.sin(np.pi * mesh.X2)
        * np.sin(np.pi * mesh.X3)
    ).to(mesh.device)

    solver_config = {
        "method": "cg",
        "tol": 1e-5,
        "max_it": 100,
        "report": True,
    }
    var = fdm_poisson(var, rhs, solver_config)

    sol = var.VAR

    # Check Dirichlet BCs
    np.testing.assert_almost_equal(
        torch.linalg.norm(sol[1:-1, 1:-1, 0] - bc_vals[0][0])
        .detach()
        .cpu()
        .numpy(),
        0,
    )
    np.testing.assert_almost_equal(
        torch.linalg.norm(sol[:, 0, :] - bc_vals[4][0]).detach().cpu().numpy(),
        0,
    )
    np.testing.assert_almost_equal(
        torch.linalg.norm(sol[:, -1, :] - bc_vals[5][0])
        .detach()
        .cpu()
        .numpy(),
        0,
    )

    # Check Neumann BC
    grad = 2 * DX[2] + sol[:, :, -2]
    grad[:, 0] = bc_vals[4][0]
    grad[:, -1] = bc_vals[5][0]
    np.testing.assert_almost_equal(
        torch.linalg.norm(sol[:, :, -1] - grad).detach().cpu().numpy(), 0
    )

    # Check Symmetry BC
    np.testing.assert_almost_equal(
        torch.linalg.norm(sol[0, :, :] - sol[1, :, :]).detach().cpu().numpy(),
        0,
    )
    # Check Periodic BC
    np.testing.assert_almost_equal(
        torch.linalg.norm(sol[-1, :, :] - sol[0, :, :]).detach().cpu().numpy(),
        0,
    )


def test_poisson_1() -> None:
    r"""Test the Poisson solver.

    Reference:
        - Zhi Shi et al (2012) (https://doi.org/10.1016/j.apm.2011.11.078)

    .. math::

        \nabla^2 p = sin(\pi x)sin(\pi y)sin(\pi z)

    And the exact solution is given by

    ..math::
        p_e = -\frac{1}{3\pi^2}sin(\pi x)sin(\pi y)sin(\pi z)

    """

    bc_types = [
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
    ]
    bc_vals = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    var = scalar_field_for_solver_test(bc_types, bc_vals)
    mesh = mesh_for_solver_test()

    # RHS of the Poisson equation
    rhs = torch.from_numpy(
        np.sin(np.pi * mesh.X1)
        * np.sin(np.pi * mesh.X2)
        * np.sin(np.pi * mesh.X3)
    ).to(mesh.device)

    # Get nalytic solution
    p_ex = (
        -1.0
        / (3 * np.pi**2)
        * np.sin(np.pi * mesh.X1)
        * np.sin(np.pi * mesh.X2)
        * np.sin(np.pi * mesh.X3)
    )

    solver_config = {
        "method": "jacobi",
        "omega": 2 / 3,
        "tol": 1e-5,
        "max_it": 1000,
        "report": True,
    }

    var = fdm_poisson(var, rhs, solver_config)

    sol = var.VAR

    np.testing.assert_almost_equal(p_ex, sol, 2)

    # Set as zero
    var.set_var_scalar(0.0)

    solver_config = {
        "method": "cg",
        "tol": 1e-5,
        "max_it": 10,
        "report": True,
    }

    var = fdm_poisson(var, rhs, solver_config)

    sol = var.VAR

    np.testing.assert_almost_equal(p_ex, sol, 2)


def test_fvc():
    """Generic way of solving FDM."""

    bc_types = [
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
    ]
    bc_vals = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    var = scalar_field_for_solver_test(bc_types, bc_vals)
    mesh = mesh_for_solver_test()

    # Create VAR
    rhs = Variables(
        name="rhs",
        type="Scalar",
        dim=1,
        objs=None,
        bc_config=None,
        x=X,
        NX=NX,
        DX=DX,
        device=DEVICE,
    )

    # RHS of the Poisson equation
    RHS = np.asarray(
        np.sin(np.pi * mesh.X1)
        * np.sin(np.pi * mesh.X2)
        * np.sin(np.pi * mesh.X3),
        dtype=np.float64,
    )

    rhs.set_var_matrix(RHS)

    # Get nalytic solution
    p_ex = (
        -1.0
        / (3 * np.pi**2)
        * np.sin(np.pi * mesh.X1)
        * np.sin(np.pi * mesh.X2)
        * np.sin(np.pi * mesh.X3)
    )

    solver_config = {
        "fvc": {
            "method": "cg",
            "tol": 1e-5,
            "max_it": 10,
            "report": True,
        },
    }

    fvc = Solver(solver_config).fvc

    # How to implement fvc.laplacian(var) + fvc.divergence(var) == rhs ?
    fvc.solve(fvc.laplacian(var) == rhs)
    # or just (fvc.laplacian(var) == rhs)

    sol = var.VAR

    np.testing.assert_almost_equal(p_ex, sol, 2)
