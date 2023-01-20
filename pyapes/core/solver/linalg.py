#!/usr/bin/env python3
"""Linear algebra for FDM solver.
Following methods are currently available to solve the linear system of equation:

* CG (Conjugated Gradient)
* BICGSTAB (Bi-Conjugated Gradient Stabilized)
"""
import warnings
from typing import Callable

import torch
from torch import Tensor

from pyapes.core.mesh import Mesh
from pyapes.core.mesh.tools import create_pad
from pyapes.core.mesh.tools import inner_slicer
from pyapes.core.solver.fdm import OPStype
from pyapes.core.variables import Field


def solve(
    var: Field,
    rhs: Tensor,
    Aop: Callable[[Field, dict[int, OPStype]], Tensor],
    eqs: dict[int, OPStype],
    config: dict[str, str | int | float | bool],
    mesh: Mesh,
) -> dict[str, int | float | bool]:
    r"""Solve Poisson equation on the rectangular grid.

    Warning:
        - Due to the way of its implementation, there is a minimum number
          of grid points in each direction: `min(mesh.nx) >= 3`

    Args:
        var: Variables to be solved.
        rhs: right hand side of the equation.
        config: solver configuration.
    """

    if config["method"] == "cg":
        report = cg(var, rhs, Aop, eqs, config, mesh)
    elif config["method"] == "bicgstab":
        # res, report = bicgstab(var, rhs, config, mesh)
        report = bicgstab(var, rhs, Aop, eqs, config, mesh)
    else:
        raise NotImplementedError

    return report


def cg(
    var: Field,
    rhs: Tensor,
    Aop: Callable[[Field, dict[int, OPStype]], Tensor],
    eqs: dict[int, OPStype],
    config: dict,
    mesh: Mesh,
) -> dict[str, int | float | bool]:
    """Conjugate gradient descent method."""

    # Padding for different dimensions
    pad = create_pad(mesh.dim)

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Parameter initialization
    # Initial residue
    tol = 1.0
    # Initial iterations
    itr = 0

    # Initial values
    Ad = torch.zeros_like(rhs)
    var_new = _apply_bc_otf(var, mesh).copy(name="var_new")

    slicer = inner_slicer(mesh.dim)

    # Initial residue
    # Ax - b = r
    r = torch.zeros_like(var())
    for i in range(var.dim):
        # Pad data for later BC application
        r[i] = pad(-rhs[i][slicer] + Aop(var_new, eqs)[i][slicer])

    d = var_new.copy(name="d")
    d.set_var_tensor(r.clone())

    while tol > tolerance:

        var_old = var_new.copy(name="var_old")

        # CG steps
        # Act of operational matrix in the search direction

        for i in range(var.dim):
            Ad[i] = -pad(Aop(d, eqs)[i][slicer])

        # Magnitude of the jump
        alpha = torch.nan_to_num(
            torch.sum(r * r, dim=var.mesh_axis)
            / torch.sum(d() * Ad, dim=var.mesh_axis),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Iterated solution
        var_new.set_var_tensor(var_old() + alpha * d())

        # Intermediate computation
        beta_denom = torch.sum(r * r, dim=var.mesh_axis)

        # Update residual
        r -= alpha * Ad

        # Compute beta
        beta = torch.sum(r * r, dim=var.mesh_axis) / beta_denom

        # Update search direction
        # d = r + beta * d
        d.set_var_tensor(r + beta * d())

        # Apply BCs
        # var_new = _apply_bc_otf(var_new, mesh).copy()
        var_new.set_var_tensor(_apply_bc_otf(var_new, mesh)())

        # Check validity of tolerance
        tol = _tolerance_check(var_new(), var_old())

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

    # Update variable
    var.VAR = var_new()  # type: ignore

    return res_report


def bicgstab(
    var: Field,
    rhs: Tensor,
    Aop: Callable[[Field, dict[int, OPStype]], Tensor],
    eqs: dict[int, OPStype],
    config: dict,
    mesh: Mesh,
) -> dict[str, int | float | bool]:
    """Bi-conjugated gradient stabilized method.

    References:
        - https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/bicgstab/bicgstab.py
        - https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
        - And modified to use utilize `torch` rather than `numpy`.
    """

    # Padding for different dimensions
    pad = create_pad(mesh.dim)

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Parameter initialization
    # Initial residue
    itr = 0

    var_new = _apply_bc_otf(var, mesh).copy(name="var_new")

    slicer = inner_slicer(mesh.dim, None)

    # Initial residue
    r = torch.zeros_like(var())
    for i in range(var.dim):
        # Pad data for later BC application
        r[i] = pad(-rhs[i][slicer] + Aop(var_new, eqs)[i][slicer])

    r0 = r.clone()
    v = torch.zeros_like(var())
    t = torch.zeros_like(var())

    p = var.copy(name="p")
    s = var.copy(name="s")

    rho = 1.0
    alpha = 1.0
    omega = 1.0

    rho_next = torch.sum(r * r, dim=var.mesh_axis)

    tol = torch.sqrt(rho_next.max()).item()

    finished: bool = False

    while not finished:

        beta = rho_next / rho * alpha / omega
        rho = rho_next.clone()

        # Update p in-place
        p_dummy = p().clone()
        p_dummy *= beta
        p_dummy -= beta * omega * v - r
        p.set_var_tensor(p_dummy)

        for i in range(var.dim):
            v[i] = -pad(Aop(p, eqs)[i][slicer])

        itr += 1

        alpha = torch.nan_to_num(
            rho / torch.sum(r0 * v, dim=var.mesh_axis),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        s.set_var_tensor(r - alpha * v)

        tol = _tolerance_check(r, alpha * v)

        if tol <= tolerance:
            var_new.set_var_tensor(var_new() + alpha * p())
            # Apply BCs
            var_new.set_var_tensor(_apply_bc_otf(var_new, mesh)())
            finished = True
            continue

        if itr >= max_it:
            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, RuntimeWarning)
            break

        for i in range(var.dim):
            t[i] = -pad(Aop(s, eqs)[i][slicer])

        omega = torch.nan_to_num(
            torch.sum(t * s(), dim=var.mesh_axis)
            / torch.sum(t * t, dim=var.mesh_axis),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rho_next = -omega * torch.sum(r0 * t, dim=var.mesh_axis)

        # Update residual
        r = s() - omega * t

        # Update solution in-place-ish.
        s.set_var_tensor(s() * omega)
        var_new.set_var_tensor(var_new() + s() + alpha * p())

        # Apply BCs
        var_new.set_var_tensor(_apply_bc_otf(var_new, mesh)())

        tol = torch.linalg.norm(r)

        if tol <= tolerance:
            finished = True

        if itr >= max_it:
            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, RuntimeWarning)
            break

    # Add report of the results
    if report:
        _solution_report(itr, tol, "CG")

    res_report = _write_report(itr, tol, itr < max_it)

    # Update variable
    var.VAR = var_new()  # type: ignore

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


def _solution_report(itr: int, tol: float, method: str) -> None:

    print(f"\n{method}: The solution  converged after {itr} iteration.")
    print(f"\ttolerance: {tol}")


def _write_report(itr: int, tol: float, converge: bool):

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
