#!/usr/bin/env python3
"""Test solver components."""
import numpy as np
import torch

from pyABC.core.boundaries import BC_OBJECT_FACTORY
from pyABC.core.boundaries import get_basic_domain
from pyABC.core.fields import Variables
from pyABC.core.mesh import Mesh
from pyABC.solver.fdm import fdm_grad
from pyABC.solver.fdm import fdm_laplacian
from pyABC.solver.fluxes import DimOrder
from pyABC.solver.fluxes import NormalDir
from pyABC.solver.operators import Solver
from pyABC.tools.utils import Container
from tests.test_variables import grid_setup

# Global settings
LX_MIN = [0.0, 0.0, 0.0]
LX_MAX = [1.0, 1.0, 1.0]
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


def vector_field_for_solver_test(bc_types: list, bc_vals: list) -> Variables:

    obj_config, bcs_config = get_basic_domain(LX, bc_types, bc_vals)

    # Create Patch object
    objs = []
    for i, obj in enumerate(obj_config):
        objs.append(BC_OBJECT_FACTORY[obj["type"]](obj, i))

    # Create VAR
    VAR = Variables(
        name="U",
        type="Vector",
        dim=3,
        objs=objs,
        bc_config=bcs_config,
        x=X,
        NX=NX,
        DX=DX,
        device=DEVICE,
    )

    return VAR


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


def test_grad():

    bc_types = [
        "neumann",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
    ]
    # Postive means samd direction with the boundary surface normal.
    # How can I differ scalar bc and vector bc?
    bc_vals = [[6.2], [1.0], [0.0], [1.0], [0.0], [1.0]]

    var = scalar_field_for_solver_test(bc_types, bc_vals)
    mesh = mesh_for_solver_test()

    # RHS of the Poisson equation
    val = np.asarray(
        mesh.X1**2,
        dtype=np.float64,
    )

    var.set_var_matrix(val)

    solver_config = {"fvm": None}

    fvm = Solver(solver_config).fvm

    grad = fvm.grad(var)

    target = mesh.X1 * 2

    # Check interial node
    np.testing.assert_almost_equal(
        target[1:-1, 1:-1, 1:-1],
        grad[0].numpy("x")[1:-1, 1:-1, 1:-1],
    )

    # Neumann BC - x
    # Manually calculate the grad(x**2) at bc_x at x0
    x0_bc_r = (mesh.X1[0, 0, 0] ** 2 + mesh.X1[0, 0, 1] ** 2) / 2
    x0_bc_l = 0.5 * bc_vals[0][0] * var.DX[0] + mesh.X1[0, 0, 0] ** 2
    # flux * S / V = flux / DX
    x0 = (x0_bc_r - x0_bc_l) / var.DX[0]

    np.testing.assert_almost_equal(x0, grad[0].numpy("x")[0, 0, 0])

    # Dirchlet BC
    # Manually calculate the grad(x**2) at bc_x at xn
    xn_bc_r = bc_vals[1][0]
    xn_bc_l = (mesh.X1[0, 0, -2] ** 2 + mesh.X1[0, 0, -1] ** 2) / 2
    xn = (xn_bc_r - xn_bc_l) / var.DX[0]

    np.testing.assert_almost_equal(xn, grad[0].numpy("x")[0, 0, -1])

    # Dirchlet BC - y = 0
    y0_bc_r = (mesh.X1[1, 0, 1] ** 2 + mesh.X1[1, 1, 1] ** 2) / 2
    y0_bc_l = bc_vals[4][0]
    y0 = (y0_bc_r - y0_bc_l) / var.DX[1]

    np.testing.assert_almost_equal(y0, grad[0].numpy("y")[1, 0, 1])


def test_div_laplacian():

    bc_types = [
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
    ]

    # Postive means samd direction with the boundary surface normal.
    # How can I differ scalar bc and vector bc?
    bc_vals_u = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    bc_vals_phi = [[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]

    u = vector_field_for_solver_test(bc_types, bc_vals_u)
    phi = scalar_field_for_solver_test(bc_types, bc_vals_phi)

    mesh = mesh_for_solver_test()

    # RHS of the Poisson equation
    val = np.asarray(
        mesh.X1**2 + mesh.X2**3,
        dtype=np.float64,
    )

    phi.set_var_matrix(val)

    u.set_var_matrix(np.asarray([val, val, val], dtype=np.float64))

    solver_config = {"fvm": None}
    fvm = Solver(solver_config).fvm

    # Scalar divergence
    u_phi = []
    for i in range(3):
        u_phi.append(phi() * u()[i])

    grad_1 = fdm_grad(u_phi[1], phi.NX, phi.DX)
    div_1 = fvm.div(phi, u)

    assert len(div_1) == 1

    # Discretization test
    div_x = div_1[0].numpy("x")[1:-1, 1:-1, 1:-1]
    div_x_fdm = grad_1.numpy("x")[1:-1, 1:-1, 1:-1]
    np.testing.assert_almost_equal(div_x, div_x_fdm)

    div_y = div_1[0].numpy("y")[1:-1, 1:-1, 1:-1]
    div_y_fdm = grad_1.numpy("y")[1:-1, 1:-1, 1:-1]
    np.testing.assert_almost_equal(div_y, div_y_fdm)

    # Vector divergence
    grad_2 = fdm_grad(torch.tensor(val * val), u.NX, u.DX)
    div_2 = fvm.div(u, u)

    assert len(div_2) == 3

    div_x = div_2[0].numpy("c")[1:-1, 1:-1, 1:-1]
    div_x_fdm = grad_2.numpy("c")[1:-1, 1:-1, 1:-1]

    np.testing.assert_almost_equal(div_x, div_x_fdm)

    coeffs = 2.0

    # Laplacian of scalar
    # Due to BC of fvm.grad, we use 2:-2 to check discretization accuracy
    lap_1_fdm = fdm_laplacian(torch.tensor(val * coeffs), u.NX, u.DX)
    lap_1 = fvm.laplacian(coeffs, phi)

    assert len(lap_1) == 1

    lap_1_fdm_x = lap_1_fdm.numpy("x")[2:-2, 2:-2, 2:-2]
    lap_1_x = lap_1[0].numpy("x")[2:-2, 2:-2, 2:-2]

    np.testing.assert_almost_equal(lap_1_x, lap_1_fdm_x)

    # Laplacian of vector
    lap_2 = fvm.laplacian(coeffs, u)

    assert len(lap_2) == 3

    lap_2_fdm_x = lap_1_fdm.numpy("c")[2:-2, 2:-2, 2:-2]
    lap_2_x = lap_2[0].numpy("c")[2:-2, 2:-2, 2:-2]
    __import__("pdb").set_trace()

    np.testing.assert_almost_equal(lap_2_x, lap_2_fdm_x)


def test_laplacian():
    pass


if __name__ == "__main__":
    # test_flux()
    test_grad()
    test_div_laplacian()
