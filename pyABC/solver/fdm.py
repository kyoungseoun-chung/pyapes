#!/usr/bin/env python3
"""Collection of FDM operators."""
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from rich import print as rprint

from pyABC.core.boundaries import apply_bc_fdm
from pyABC.core.fields import Variables
from pyABC.solver.fluxes import Flux


class Laplacian:
    r"""Solve for the Poisson equation.

    .. math::

        \nabla^2 \Phi = \vec{b}

    Args:
        var: Variables to be solved ($\Phi$)
        rhs: right hand size of the equation ($\vec{b}$)
    """

    def set_config(self, config: dict):

        self.config = config

    def __call__(self, var: Variables) -> Any:
        """
        Note:
            - Self return type is not available yet (PEP 673 will be available in the future)
            - Therefore, here the return type is set to `Any`
        """
        self.var = var

        return self

    def __eq__(self, rhs: Variables) -> Variables:

        return fdm_poisson(self.var, rhs.VAR, self.config)


def fdm_poisson(var: Variables, rhs: torch.Tensor, config: dict) -> Variables:
    r"""Solve Poisson equation on the rectangular grid.

    Note:
        - Due to the way of its implementation, there is a minimum number
          of grid points in each direction: `min(NX) >= 3`

    .. math::

        \frac{\partial^2 \Phi}{\partial x_i \partial x_i} = b_{ij}

    Args:
        var: Variables to be solved.
        rhs: right hand side of the equation.
        config: solver configuration.
    """

    method = config["method"]

    if method == "jacobi":
        res = _jacobi(var, rhs, config)
    elif method == "cg":
        res = _cg(var, rhs, config)
    else:
        from pyABC.tools.errors import WrongInputError

        msg = "Unsupported method!"
        raise WrongInputError(msg)

    return res


def _Av(var: torch.Tensor, DX: npt.NDArray[np.float64]) -> torch.Tensor:
    r"""Computes the action of (-$\mathbf{A}$) the Poisson operator on
    any vector $\vec{v}$ for the interior grid nodes
    Therefore, compute A used in

    .. math::
        -\mathbf{A}\vec{v}

    Args:
        var: input field
        DX: grid spacing

    Returns:
        Action of A on v
    """

    Av = -(
        (
            var[1:-1, 1:-1, :-2]
            - 2.0 * var[1:-1, 1:-1, 1:-1]
            + var[1:-1, 1:-1, 2:]
        )
        / DX[0] ** 2  # x-direction
        + (
            var[1:-1, :-2, 1:-1]
            - 2.0 * var[1:-1, 1:-1, 1:-1]
            + var[1:-1, 2:, 1:-1]
        )
        / DX[1] ** 2  # y-direction
        + (
            var[:-2, 1:-1, 1:-1]
            - 2.0 * var[1:-1, 1:-1, 1:-1]
            + var[2:, 1:-1, 1:-1]
        )
        / DX[2] ** 2  # z-direction
    )

    return Av


def _Aop(
    var: torch.Tensor, rhs: torch.Tensor, DX: npt.NDArray[np.float64]
) -> torch.Tensor:
    """Jacobian operator.

    Args:
        var: Variable to be discretized
        rhs: RHS of the equation
        DX: grid spacing
    """

    xy = DX[0] ** 2 * DX[1] ** 2
    yz = DX[1] ** 2 * DX[2] ** 2
    xz = DX[0] ** 2 * DX[2] ** 2
    xyz = DX[0] ** 2 * DX[1] ** 2 * DX[2] ** 2

    Aop = (
        (var[1:-1, 1:-1, 2:] + var[1:-1, 1:-1, :-2]) * yz
        + (var[1:-1, 2:, 1:-1] + var[1:-1, :-2, 1:-1]) * xz
        + (var[2:, 1:-1, 1:-1] + var[:-2, 1:-1, 1:-1]) * xy
        - rhs[1:-1, 1:-1, 1:-1] * xyz
    ) / (2 * xy + 2 * yz + 2 * xz)

    return Aop


def _jacobi(var: Variables, rhs: torch.Tensor, config: dict) -> Variables:
    """Jacobi method."""

    dx = var.DX

    tolerance = config["tol"]
    max_it = config["max_it"]
    # Relaxation factor
    omega = config["omega"]
    report = config["report"]

    # Initial parameters
    tol = 1.0
    itr = 0
    var_new = var.VAR.clone()

    while tol > tolerance:

        if itr > max_it:
            from pyABC.tools.errors import MaximunIterationReachedWarning

            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, MaximunIterationReachedWarning)
            break

        var_old = var_new.clone()

        var_new[1:-1, 1:-1, 1:-1] = (1 - omega) * var_new[
            1:-1, 1:-1, 1:-1
        ] + omega * _Aop(var_old, rhs, dx)

        # Apply bc
        if var.bcs is not None:
            var_new = apply_bc_fdm(var_new, var.bcs, var.masks)

        # Compute tolerance
        tol = torch.linalg.norm(var_new - var_old)

        # Check validity of tolerance
        if torch.isnan(tol) or torch.isinf(tol):
            from pyABC.tools.errors import SolutionDoesNotConverged

            msg = f"Invalid tolerance detected! tol: {tol}"
            raise SolutionDoesNotConverged(msg)

        # Add iteration counts
        itr += 1

    else:

        # Add report of the results
        if report:
            rprint(
                f"\n[bold green]Jacobi[/bold green] : [bold yellow]The solution  converged after [bold red]{itr}[/bold red] iteration. [/bold yellow]"
            )
            rprint(
                f"\t[bold green]tolerance[/bold green]: [bold blue]{tol}[/bold blue]"
            )

    # Update variable
    var.VAR = var_new

    return var


def _cg(var: Variables, rhs: torch.Tensor, config: dict) -> Variables:
    """Conjugate gradient descent method."""

    # Grid information
    dx = var.DX

    tolerance = config["tol"]
    max_it = config["max_it"]
    report = config["report"]

    # Paramter initialization
    # Initial residue
    tol = 1.0
    # Initial iterations
    itr = 0

    # Initial values
    r = torch.zeros_like(rhs)
    Ad = torch.zeros_like(rhs)
    var_new = var.VAR.clone()
    var_old = var.VAR.clone()

    # Initial residue
    r[1:-1, 1:-1, 1:-1] = -rhs[1:-1, 1:-1, 1:-1] - _Av(var_old, dx)
    d = r.clone()

    while tol > tolerance:

        if itr > max_it:
            from pyABC.tools.errors import MaximunIterationReachedWarning

            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {max_it}"
            warnings.warn(msg, MaximunIterationReachedWarning)
            break

        # CG steps
        # Laplacian of the search direction, BC of r?
        Ad[1:-1, 1:-1, 1:-1] = _Av(d, dx)

        # Magnitude of the jump
        alpha = torch.sum(r * r) / torch.sum(d * Ad)

        # Iterated solution
        var_new = var_old + alpha * d

        # Intermediate computation
        beta_denom = torch.sum(r * r)

        # Update residual
        r = r - alpha * Ad

        # Compute beta
        beta = torch.sum(r * r) / beta_denom

        # Update search direction
        d = r + beta * d

        # Apply bc
        if var.bcs is not None:
            var_new = apply_bc_fdm(var_new, var.bcs, var.masks)

        tol = torch.linalg.norm(var_new - var_old)
        # Check validity of tolerance
        if torch.isnan(tol) or torch.isinf(tol):
            from pyABC.tools.errors import SolutionDoesNotConverged

            msg = f"Invalid tolerance detected! tol: {tol}"
            raise SolutionDoesNotConverged(msg)

        var_old = var_new.clone()

        # Add iteration counts
        itr += 1

    else:

        # Add report of the results
        if report:
            rprint(
                f"\n[bold green]CG[/bold green] : [bold yellow]The solution  converged after [bold red]{itr}[/bold red] iteration. [/bold yellow]"
            )
            rprint(
                f"\t[bold green]tolerance[/bold green]: [bold blue]{tol}[/bold blue]"
            )

    # Update variable
    var.VAR = var_new

    return var


def fdm_laplacian(
    var: torch.Tensor, nx: npt.NDArray[np.int64], dx: npt.NDArray[np.float64]
) -> Flux:
    """Second derivatives based on volume center.
    Use a central difference with a second-order accuracy.
    Boundaries are calculated by back/forward differences.

    Args:
        var: variable tensor to be discretized.
        nx: number of grid points
        dx: grid spacing

    """

    flux = Flux(nx, dx)

    # FDM
    flux.x = (torch.roll(var, -1, 2) - 2 * var + torch.roll(var, 1, 2)) / (
        dx[0] ** 2
    )
    flux.y = (torch.roll(var, -1, 1) - 2 * var + torch.roll(var, 1, 1)) / (
        dx[1] ** 2
    )
    flux.z = (torch.roll(var, -1, 0) - 2 * var + torch.roll(var, 1, 0)) / (
        dx[2] ** 2
    )

    # Boundary treatment
    flux.x[:, :, 0] = (
        2 * var[:, :, 0] - 5 * var[:, :, 1] + 4 * var[:, :, 2] - var[:, :, 3]
    ) / (dx[0] ** 2)
    flux.x[:, :, -1] = (
        2 * var[:, :, -1]
        - 5 * var[:, :, -2]
        + 4 * var[:, :, -3]
        - var[:, :, -4]
    ) / (dx[0] ** 2)

    flux.y[:, 0, :] = (
        2 * var[:, 0, :] - 5 * var[:, 1, :] + 4 * var[:, 2, :] - var[:, 3, :]
    ) / (dx[1] ** 2)
    flux.y[:, -1, :] = (
        2 * var[:, -1, :]
        - 5 * var[:, -2, :]
        + 4 * var[:, -3, :]
        - var[:, -4, :]
    ) / (dx[1] ** 2)

    flux.z[0, :, :] = (
        2 * var[0, :, :] - 5 * var[1, :, :] + 4 * var[2, :, :] - var[3, :, :]
    ) / (dx[2] ** 2)
    flux.z[-1, :, :] = (
        2 * var[-1, :, :]
        - 5 * var[-2, :, :]
        + 4 * var[-3, :, :]
        - var[-4, :, :]
    ) / (dx[2] ** 2)

    flux.sum()

    return flux


def fdm_grad(
    var: torch.Tensor, nx: npt.NDArray[np.int64], dx: npt.NDArray[np.float64]
) -> Flux:
    """Resuable finite difference operator for the first derivative.
    Use a central difference with a second-order accuracy.
    Boundaries are calculated by back/forward differences.

    Args:
        var: variable tensor to be discretized.
        nx: number of grid points
        dx: grid spacing
    """

    flux = Flux(nx, dx)

    # FDM
    flux.x = (torch.roll(var, -1, 2) - torch.roll(var, 1, 2)) / (2 * dx[0])
    flux.y = (torch.roll(var, -1, 1) - torch.roll(var, 1, 1)) / (2 * dx[1])
    flux.z = (torch.roll(var, -1, 0) - torch.roll(var, 1, 0)) / (2 * dx[2])

    # Boundary treat
    # x-direction
    flux.x[:, :, 0] = (-3 * var[:, :, 0] + 4 * var[:, :, 1] - var[:, :, 2]) / (
        2 * dx[0]
    )
    flux.x[:, :, -1] = (
        3 * var[:, :, -1] - 4 * var[:, :, -2] + var[:, :, -3]
    ) / (2 * dx[0])

    # y-direction
    flux.y[:, 0, :] = (-3 * var[:, 0, :] + 4 * var[:, 1, :] - var[:, 2, :]) / (
        2 * dx[1]
    )
    flux.y[:, -1, :] = (
        3 * var[:, -1, :] - 4 * var[:, -2, :] + var[:, -3, :]
    ) / (2 * dx[1])

    # x-direction
    flux.z[0, :, :] = (-3 * var[0, :, :] + 4 * var[1, :, :] - var[2, :, :]) / (
        2 * dx[2]
    )
    flux.z[-1, :, :] = (
        3 * var[-1, :, :] - 4 * var[-2, :, :] + var[-3, :, :]
    ) / (2 * dx[2])

    flux.sum()

    return flux
