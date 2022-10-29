#!/usr/bin/env python3
"""Linear algebra for FVM solver.
Following methods are currently available to solve the linear system of equation:

* CG (Conjugated Gradient)
* BICGSTAB(Bi-Conjugated Gradient Stabilized) <- WIP
"""
import warnings
from typing import Union

import torch
from torch import Tensor

from pyapes.core.mesh import Mesh
from pyapes.core.solver.tools import create_pad
from pyapes.core.solver.tools import inner_slicer
from pyapes.core.variables import Field
from pyapes.core.variables import Flux


def solve(
    var: Field,
    rhs: Tensor,
    config: dict,
    mesh: Mesh,
    ops: dict[int, dict[str, Union[Flux, str]]],
) -> tuple[Field, dict]:
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
        res, report = cg(var, rhs, ops, config, mesh)
    elif config["method"] == "bicgstab":
        res, report = bicgstab(var, rhs, ops, config, mesh)
    else:
        raise NotImplementedError

    return res, report


# WIP: Need implementation!
def _Aop(ops: dict[int, dict[str, Union[Flux, str]]]) -> Tensor:
    pass


def cg(
    var: Field,
    rhs: Tensor,
    ops: dict[int, dict[str, Union[Flux, str]]],
    config: dict,
    mesh: Mesh,
) -> tuple[Field, dict]:
    """Conjugate gradient descent method."""

    # Padding for different dimensions
    pad = create_pad(mesh.dim)

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Paramter initialization
    # Initial residue
    tol = 1.0
    # Initial iterations
    itr = 0

    # Initial values
    Ad = torch.zeros_like(rhs)
    var_new = _apply_bc_otf(var, var(), mesh).clone()

    slicer = inner_slicer(mesh.dim)

    # Initial residue
    r = pad(-rhs[slicer] - _Aop(ops))
    d = r.clone()

    while tol > tolerance:

        var_old = var_new.clone()
        # CG steps
        # Laplacian of the search direction
        Ad = pad(_Aop(ops))

        # Magnitude of the jump
        alpha = torch.sum(r * r) / torch.sum(d * Ad)

        # Iterated solution
        var_new[0] = var_old[0] + alpha * d

        # Intermediate computation
        beta_denom = torch.sum(r * r)

        # Update residual
        r -= alpha * Ad

        # Compute beta
        beta = torch.sum(r * r) / beta_denom

        # Update search direction
        d = r + beta * d

        # Apply BCs
        var_new = _apply_bc_otf(var, var_new, mesh).clone()

        # Check validity of tolerance
        tol = _tolerance_check(var_new, var_old)

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
    var.VAR = var_new  # type: ignore

    return var, res_report


# WIP: NOT YET VALIDATED!
def bicgstab(
    var: Field,
    rhs: Tensor,
    ops: dict[int, dict[str, Union[Flux, str]]],
    config: dict,
    mesh: Mesh,
) -> tuple[Field, dict]:
    """Bi-conjugated gradient stabilized method.

    Referenced from
    - https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/bicgstab/bicgstab.py
    - https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    And modified to use with torch rather than numpy.
    """

    # Padding for different dimensions
    pad = create_pad(mesh.dim)

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Paramter initialization
    # Initial residue
    itr = 0

    var_new = _apply_bc_otf(var, var(), mesh).clone()

    slicer = inner_slicer(mesh.dim)

    # Initial residue
    r = pad(-rhs[slicer] - _Aop(ops))
    r0 = r.clone()
    v = torch.zeros_like(rhs)[slicer]
    p = torch.zeros_like(v)

    rho = 1.0
    alpha = 1.0
    omega = 1.0

    rho_next = torch.sum(r * r)
    tol = torch.sqrt(rho_next).item()

    finished: bool = False

    while not finished:

        beta = rho_next / rho * alpha / omega
        rho = rho_next

        # Update p in-place
        p *= beta
        p -= beta * omega * v - r

        v = _Aop(ops) * p
        itr += 1

        alpha = rho / torch.sum(r0 * v)
        s = r - alpha * v

        tol = torch.linalg.norm(s)

        if tol <= tolerance:
            var_new += alpha * p
            finished = True
            continue

        if itr >= max_it:
            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, RuntimeWarning)
            break

        t = _Aop(ops) * s
        omega = torch.sum(t * s) / torch.sum(t * t)
        rho_next = -omega * torch.sum(r0 * t)

        # Update residual
        r = s - omega * t

        # Update solution in-place-ish. Note that 'z *= omega' alters s if
        # precon = None. That's ok since s is no longer needed in this iter.
        # 'q *= alpha' would alter p.
        s *= omega
        var_new += s + alpha * p

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
    var.VAR = var_new  # type: ignore

    return var, res_report


def _apply_bc_otf(var: Field, var_t: Tensor, mesh: Mesh) -> Tensor:
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
        for bc, m in zip(var.bcs, var.masks):
            mask = var.masks[m]
            var_t = bc.apply(mask, var_t, mesh)

    return var_t


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

    tol = torch.linalg.norm(var_new - var_old)

    # Check validity of tolerance
    if torch.isnan(tol) or torch.isinf(tol):
        msg = f"Invalid tolerance detected! tol: {tol}"
        raise RuntimeError(msg)

    return tol.item()
