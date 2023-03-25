#!/usr/bin/env python3
"""Linear algebra for FDM solver.
Following methods are currently available to solve the linear system of equation:

* CG (Conjugated Gradient)
* BICGSTAB (Bi-Conjugated Gradient Stabilized)
"""
import warnings
from typing import Callable
from typing import TypedDict

import torch
from torch import Tensor

from pyapes.mesh import Mesh
from pyapes.mesh.tools import boundary_slicer
from pyapes.solver.fdm import OPStype
from pyapes.solver.tools import FDMSolverConfig
from pyapes.variables import Field


class ReportType(TypedDict):
    """Report contains solver information."""

    itr: int
    """Total number of iterations until convergence."""
    tol: float
    """Convergence tolerance."""
    converge: bool
    """Result of convergence. If `itr<max_it`, then `converge=True`."""


def solve(
    var: Field,
    rhs: Tensor,
    Aop: Callable[[Field, dict[int, OPStype]], Tensor],
    eqs: dict[int, OPStype],
    config: FDMSolverConfig,
    mesh: Mesh,
) -> ReportType:
    r"""Solve Poisson equation on the rectangular grid.

    Warning:
        - Due to the way of its implementation, there is a minimum number
          of grid points in each direction: `min(mesh.nx) >= 3`

    Args:
        var: Variables to be solved.
        rhs: right hand side of the equation.
        config: solver configuration.
    """

    method = config["method"]

    assert (
        isinstance(method, str) and method is not None
    ), "Linalg: solver method is not defined!"

    # Just to make it sure that the method is in lower case
    method = method.lower()

    if method == "cg":
        report = cg(var, rhs, Aop, eqs, config, mesh)
    elif method == "bicgstab":
        report = bicgstab(var, rhs, Aop, eqs, config, mesh)
    else:
        raise RuntimeError(
            f"Linalg: solver only supports CG and BICGSTAB. {method=} would be a typo or is not supported."
        )

    return report


def cg(
    var: Field,
    rhs: Tensor,
    Aop: Callable[[Field, dict[int, OPStype]], Tensor],
    eqs: dict[int, OPStype],
    config: FDMSolverConfig,
    mesh: Mesh,
) -> ReportType:
    """Conjugate gradient descent method."""

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Parameter initialization
    # Initial residue
    tol = 1.0
    # Initial iterations
    itr = 0

    slicer = boundary_slicer(mesh.dim, var.bcs)

    # Initial values
    _apply_bc_otf(var, mesh)
    Ad = torch.zeros_like(rhs)

    # Initial residue
    # Ax - b = r
    r = var.zeros_like_tensor()
    for i in range(var.dim):
        r[i][slicer] = rhs[i][slicer] - Aop(var, eqs)[i][slicer]

    d = var.copy(name="d")
    d.set_var_tensor(r.clone())

    while tol > tolerance:
        var.save_old()

        # CG steps
        # Act of operational matrix in the search direction
        for i in range(var.dim):
            Ad[i][slicer] = Aop(d, eqs)[i][slicer]

        # Magnitude of the jump
        alpha = _nan_to_num(
            torch.sum(r * r, dim=var.mesh_axis) / torch.sum(d() * Ad, dim=var.mesh_axis)
        )
        # Iterated solution
        var.set_var_tensor(var() + alpha * d())

        # Apply BCs
        _apply_bc_otf(var, mesh)

        # Intermediate computation
        beta_denom = torch.sum(r * r, dim=var.mesh_axis)

        # Update residual
        r -= alpha * Ad

        # Check validity of tolerance
        tol = _tolerance_check(var(), var.VARo)

        # Compute beta
        beta = torch.sum(r * r, dim=var.mesh_axis) / beta_denom

        # Update search direction
        # d = r + beta * d
        d.set_var_tensor(r + beta * d())

        # Add iteration counts
        itr += 1

        if itr > max_it:
            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, RuntimeWarning)
            break

    else:
        # Add report of the results
        if report:
            _solution_report(itr, tol, "CG")

    res_report = _write_report(itr, tol, itr < max_it)

    return res_report


def bicgstab(
    var: Field,
    rhs: Tensor,
    Aop: Callable[[Field, dict[int, OPStype]], Tensor],
    eqs: dict[int, OPStype],
    config: FDMSolverConfig,
    mesh: Mesh,
) -> ReportType:
    """Bi-conjugated gradient stabilized method.

    References:
        - https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/bicgstab/bicgstab.py
        - https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
        - And modified to use utilize `torch` rather than `numpy`.
    """

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Parameter initialization
    # Initial residue
    itr = 0

    slicer = boundary_slicer(mesh.dim, var.bcs)

    _apply_bc_otf(var, mesh)

    # Initial residue
    r0 = var.zeros_like_tensor()
    for i in range(var.dim):
        r0[i][slicer] = rhs[i][slicer] - Aop(var, eqs)[i][slicer]

    r = r0.clone()
    t = var.zeros_like_tensor()
    v = var.zeros_like_tensor()
    p = var.zeros_like(name="p")
    s = var.zeros_like(name="s")

    rho = 1.0
    alpha = 1.0
    omega = 1.0
    rho_next = torch.sum(r0 * r0, dim=var.mesh_axis)

    tol = torch.sqrt(rho_next.max()).item()

    finished: bool = False

    while not finished:
        var.save_old()
        beta = rho_next / rho * alpha / omega

        rho = rho_next

        # Update p in-place
        p.set_var_tensor(r + beta * (p() - omega * v))

        for i in range(var.dim):
            v[i][slicer] = Aop(p, eqs)[i][slicer]

        itr += 1

        # alpha = rho / dot(r0, v)
        alpha = _nan_to_num(rho / torch.sum(r0 * v, dim=var.mesh_axis))

        if torch.isnan(v).sum() > 0:
            pass

        s.set_var_tensor(r - alpha * v)

        # Check tolerance
        tol = _tolerance_check(r, alpha * v)

        if tol <= tolerance:
            var.set_var_tensor(var() + alpha * p())
            # Apply BCs
            _apply_bc_otf(var, mesh)
            finished = True
            continue

        for i in range(var.dim):
            t[i][slicer] = Aop(s, eqs)[i][slicer]

        # omega dot(t, s) / dot(t, t)
        omega = _nan_to_num(
            torch.sum(t * s(), dim=var.mesh_axis) / torch.sum(t * t, dim=var.mesh_axis)
        )

        rho_next = -omega * torch.sum(r0 * t, dim=var.mesh_axis)

        # Update solution in-place-ish.
        var.set_var_tensor(var() + alpha * p() + s() * omega)

        # Apply BCs
        _apply_bc_otf(var, mesh)

        # Update residual
        r = s() - omega * t

        # Check tolerance
        tol = _tolerance_check(s(), omega * t)

        if tol <= tolerance:
            finished = True

        # Check maximum number of iterations
        if itr >= max_it:
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, RuntimeWarning)
            break

    # Add report of the results
    if report:
        _solution_report(itr, tol, "BICGSTAB")

    res_report = _write_report(itr, tol, itr < max_it)

    return res_report


def _apply_bc_otf(var: Field, mesh: Mesh) -> Field:
    """Apply bc on the fly (OTF)."""

    # Apply BCs
    if len(var.bcs) > 0:
        if mesh.obstacle is not None:
            # Fill zero to inner obstacle
            # Loop over inner object and assign values
            # for m, v in zip(var.mask_inner, var.obj_inner_interp):
            #     var_t[0][var.mask_inner[m]] = v
            raise NotImplementedError

        # Apply BC
        for d in range(var.dim):
            for bc in var.bcs:
                bc.apply(var(), mesh.grid, d)

    return var


def _nan_to_num(t_in: Tensor) -> Tensor:
    """Convert any Nan values in `input` Tensor to zero."""

    return torch.nan_to_num(t_in, nan=0.0, posinf=0.0, neginf=0.0)


def _solution_report(itr: int, tol: float, method: str) -> None:
    """Report result of the solver with total number of iterations, solution tolerance."""

    print(f"\n{method}: The solution  converged after {itr} iteration.")
    print(f"\ttolerance: {tol}")


def _write_report(itr: int, tol: float, converge: bool) -> ReportType:
    """Write report of the solver contains total number of iterations, solution tolerance, and convergence."""

    return {"itr": itr, "tol": tol, "converge": converge}


def _tolerance_check(var_new: Tensor, var_old: Tensor) -> float:
    """Check tolerance

    Raise:
        RuntimeError: if unrealistic value detected.
    """
    dim = var_new.shape[0]

    tol = torch.zeros(dim, dtype=var_new.dtype)
    for d in range(dim):
        tol[d] = torch.linalg.norm(var_new[d] - var_old[d])

    # Check validity of tolerance
    if torch.isnan(tol) or torch.isinf(tol):
        msg = f"Invalid tolerance detected! tol: {tol}"
        raise RuntimeError(msg)

    return torch.max(tol).item()
