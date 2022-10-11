#!/usr/bin/env python3
import warnings
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Union

import torch
from rich import print as rprint
from torch import Tensor
from torch.nn import ConstantPad1d
from torch.nn import ConstantPad2d
from torch.nn import ConstantPad3d

from pyABC.core.mesh import Mesh
from pyABC.core.variables import Field


class Discretizer(ABC):
    """Base class for operators."""

    def set_config(self, config: dict):

        self.config = config

    def __call__(self, var: Field) -> Any:
        """
        Note:
            - Self return type is not available yet (PEP 673 will be available in the future)
            - Therefore, here the return type is set to `Any`
        """
        self.var = var

        return self

    def __eq__(self, rhs: Tensor or float) -> Any:

        if type(rhs) is float:
            self.rhs = torch.zeros_like(self.var())[0] + rhs
        else:
            self.rhs = rhs

        return self

    @abstractmethod
    def solve(self) -> Field:
        """Abstract method for solving the given PDE."""
        pass


class Custom(Discretizer):
    def solve(self) -> Field:
        """Solve for the poisson equation."""

        res, self.report = fdm_op(
            self.var, self.rhs, self.config, self.var.mesh
        )

        return res


class Laplacian(Discretizer):
    r"""Operator to solve the Poisson equation.

    Note:
        - Only works for the scalar field

    .. math::

        \nabla^2 \Phi = \vec{b}

    Args:
        var: Variables to be solved ($\Phi$)
        rhs: right hand size of the equation ($\vec{b}$)
    """

    var: Field
    rhs: Tensor

    def solve(self) -> Field:
        """Solve for the poisson equation."""

        res, self.report = fdm_op(
            self.var, self.rhs, self.config, self.var.mesh
        )

        return res


def fdm_op(
    var: Field, rhs: Tensor, config: dict, mesh: Mesh
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

    method = config["method"]

    if method == "jacobi":
        Aop = config["Aop"] if "Aop" in config else _Ajacobi
        res, report = _jacobi(var, rhs, Aop, config, mesh)
    elif method == "gs":
        Aop = config["Aop"] if "Aop" in config else _Ags
        res, report = _gs(var, rhs, Aop, config, mesh)
    elif method == "cg":
        Aop = config["Aop"] if "Aop" in config else _Acg
        res, report = _cg(var, rhs, Aop, config, mesh)
    elif method == "cg_newton":
        Aop = config["Aop"] if "Aop" in config else _Acg
        res, report = _cg_newton(var, rhs, Aop, config, mesh)
    else:
        msg = "Unsupported method!"
        raise ValueError(msg)

    return res, report


def _Ags(
    var_o: Tensor, var_n: Tensor, rhs: Tensor, DX: Tensor, dim: int
) -> Tensor:
    r"""Gauss Seidel operator for the Poisson equation.

    Args:
        var_o: old variable to be used
        var_n: new variable to be used
        rhs: RHS of the equation
        DX: grid spacing
        dim: direction to apply the operation
    """

    slicer = inner_slicer(dim)

    ddx = 0
    Aop = torch.zeros_like(var_o)

    for i in range(dim):

        ddx += 2.0 / (DX[i] ** 2)
        Aop += (torch.roll(var_n, 1, i) + torch.roll(var_o, -1, i)) / DX[
            i
        ] ** 2

    return (Aop - rhs)[slicer] / ddx


def _Acg(var: Tensor, DX: Tensor, dim: int) -> Tensor:
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

    slicer = inner_slicer(dim)
    Acg = torch.zeros_like(var)

    # Second order FDM
    for i in range(dim):
        Acg -= (
            torch.roll(var, -1, i) - 2.0 * var + torch.roll(var, 1, i)
        ) / DX[i] ** 2

    return Acg[slicer]


def _Ajacobi(var: Tensor, rhs: Tensor, DX: Tensor, dim: int) -> Tensor:
    r"""Jacobian operator for the Poisson equation.

        .. math::

            \nabla^2 \Phi = \vec{b}

    For example, in 2D case, the finite difference discretization gives

        .. math::

            \frac{\Phi^{i+1, j} - 2\Phi^{i, j} + \Phi^{i-1, j}}{\Delta i^2}
            + \frac{\Phi^{i, j+1} - 2\Phi^{i, j} + \Phi^{i, j-1}}{\Delta j^2}
            = b^{i, j}.

    If we organize above equation for the Jacobi method,

        .. math::

            2\frac{\Phi^{i, j}}{\Delta_i^2} + 2\frac{\Phi^{i, j}}{\Delta_j^2}
            = \frac{\Phi^{i+1, j} + \Phi^{i-1, j}}{\Delta_i^2}
            + \frac{\Phi^{i, j+1} + \Phi^{i, j-1}}{\Delta_j^2}
            - b^{i, j}.

    Therefore, the final equation becomes

        .. math::

            \Phi^{i, j}
            = \left( \frac{2}{\Delta_i^2} + \frac{2}{\Delta_j^2} \right)^{-1}
            \frac{\Phi^{i+1, j} + \Phi^{i-1, j}}{\Delta_i^2}
            + \left( \frac{2}{\Delta_i^2} + \frac{2}{\Delta_j^2}  \right)^{-1}
            \frac{\Phi^{i, j+1} + \Phi^{i, j-1}}{\Delta_j^2}
            - \left( \frac{2}{\Delta_i^2} + \frac{2}{\Delta_j^2}  \right)^{-1} b^{i, j}.

    Args:
        var: Variable to be discretized
        rhs: RHS of the equation
        DX: grid spacing
        dim: direction to apply the operation
    """

    slicer = inner_slicer(dim)

    ddx = 0
    Aj = torch.zeros_like(var)

    for i in range(dim):

        ddx += 2.0 / (DX[i] ** 2)
        Aj += (torch.roll(var, -1, i) + torch.roll(var, 1, i)) / DX[i] ** 2

    return (Aj - rhs)[slicer] / ddx


def _gs(
    var: Field, rhs: Tensor, Aop: Callable, config: dict, mesh: Mesh
) -> tuple[Field, dict]:
    """Gauss-Seidel method."""
    # Padding for different dimensions
    pad = _create_pad(mesh.dim)

    dx = var.nx

    tolerance = config["tol"]
    max_it = config["max_it"]
    # Relaxation factor
    omega = config["omega"]
    report = config["report"]

    rhs_0 = rhs.clone()

    try:
        rhs_func = config["rhs"]
        rhs = _apply_rhs_otf(rhs_func, var(), rhs_0)

    except KeyError:
        rhs_func = None

    # Initial parameters
    tol = 1.0
    itr = 0
    var_new = _apply_bc_otf(var, var(), mesh).clone()

    while tol > tolerance:

        var_old = var_new.clone()

        var_new[0] = (1 - omega) * var_new[0] + omega * pad(
            Aop(var_old[0], var_new[0], rhs, dx, mesh.dim)
        )

        # Apply bc
        var_new = _apply_bc_otf(var, var_new, mesh)

        # Compute new RHS
        if rhs_func is not None:
            rhs = _apply_rhs_otf(rhs_func, var_new, rhs_0).clone()

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
            _solution_report(itr, tol, "GS")

    res_report = _write_report(itr, tol, itr < max_it)

    # Update variable << TYPING IS A BIT WEIRD. JUST IGNORE NOW
    var.VAR = var_new  # type: ignore

    return var, res_report


def _jacobi(
    var: Field, rhs: Tensor, Aop: Callable, config: dict, mesh: Mesh
) -> tuple[Field, dict]:
    """Jacobi method."""
    # Padding for different dimensions
    pad = _create_pad(mesh.dim)
    dx = var.dx

    tolerance = config["tol"]
    max_it = config["max_it"]

    # Relaxation factor
    omega = config["omega"]
    report = config["report"]

    rhs_0 = rhs.clone()

    try:
        rhs_func = config["rhs"]
        rhs = _apply_rhs_otf(rhs_func, var(), rhs_0)

    except KeyError:
        rhs_func = None

    # Initial parameters
    tol = 1.0
    itr = 0
    var_new = _apply_bc_otf(var, var(), mesh).clone()

    while tol > tolerance:

        var_old = var_new.clone()

        var_new[0] = (1 - omega) * var_new[0] + omega * pad(
            Aop(var_old[0], rhs, dx, mesh.dim)
        )

        # Apply bc
        var_new = _apply_bc_otf(var, var_new, mesh)

        # Check tolerance
        tol = _tolerance_check(var_new, var_old)

        # Compute new RHS
        if rhs_func is not None:
            rhs = _apply_rhs_otf(rhs_func, var_new, rhs_0)

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
            _solution_report(itr, tol, "Jacobi")

    res_report = _write_report(itr, tol, itr < max_it)

    var.VAR = var_new  # type: ignore

    return var, res_report


def _cg_newton(
    var: Field, rhs: Tensor, Aop: Callable, config: dict, mesh: Mesh
) -> tuple[Field, dict]:
    """Conjugate gradient descent method with varying rhs."""

    newton_tol = config["newton_tol"]
    newton_max_it = config["newton_max_it"]
    newton_report = config["newton_report"]

    res_report = {}

    tol = 1.0
    itr = 0
    var_old = var().clone()

    while tol < newton_tol:

        var, res_report = _cg(var, rhs, Aop, config, mesh)

        var_new = var().clone()

        tol = _tolerance_check(var_new, var_old)

        # Add iteration counts
        itr += 1

        if itr > newton_max_it:
            # Convergence is not achieved
            msg = f"Maximum iteration reached! max_it: {newton_max_it}"
            warnings.warn(msg, RuntimeWarning)
            break

    else:

        # Add report of the results
        if newton_report:
            _solution_report(itr, tol, "CG_NEWTON")

    res_report = _write_report(itr, tol, itr < newton_max_it)

    return var, res_report


def _cg(
    var: Field, rhs: Tensor, Aop: Callable, config: dict, mesh: Mesh
) -> tuple[Field, dict]:
    """Conjugate gradient descent method."""

    # Padding for different dimensions
    pad = _create_pad(mesh.dim)

    # Grid information
    dx = var.dx

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
    r = pad(-rhs[slicer] - Aop(var_new[0], dx, mesh.dim))
    d = r.clone()

    while tol > tolerance:

        var_old = var_new.clone()
        # CG steps
        # Laplacian of the search direction
        Ad = pad(Aop(d, dx, mesh.dim))

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


def _apply_rhs_otf(rhs_func: Callable, var_t: Tensor, rhs_0: Tensor) -> Tensor:
    """Recompute RHS on the fly (OTF)."""

    # Compute new RHS
    # New RHS based on var_old
    rhs_from_func = rhs_func(var_t)

    return (rhs_0 + rhs_from_func[0]).clone()


def _apply_bc_otf(var: Field, var_t: Tensor, mesh: Mesh) -> Tensor:
    """Apply bc on the fly (OTF)."""

    # Apply BCs
    if len(var.bcs) > 0:

        if mesh.obstacle is not None:
            # Fill zero to inner obstacle
            # Loop over inner object and assign values
            for m, v in zip(var.mask_inner, var.obj_inner_interp):
                var_t[0][var.mask_inner[m]] = v

        # Apply BC
        for bc, m in zip(var.bcs, var.masks):
            mask = var.masks[m]
            var_t = bc.apply(mask, var_t, mesh)

    return var_t


def _solution_report(itr: int, tol: float, method: str) -> None:

    rprint(
        f"\n[bold green]{method}[/bold green] : [bold yellow]The solution  converged after [bold red]{itr}[/bold red] iteration. [/bold yellow]"
    )
    rprint(
        f"\t[bold green]tolerance[/bold green]: [bold blue]{tol}[/bold blue]"
    )


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


def _create_pad(
    dim: int,
) -> Union[ConstantPad1d, ConstantPad2d, ConstantPad3d]:
    """Create padd object."""

    if dim == 1:
        return ConstantPad1d(1, 0)
    elif dim == 2:
        return ConstantPad2d(1, 0)
    else:
        return ConstantPad3d(1, 0)


def inner_slicer(dim: int) -> tuple:

    if dim == 1:
        return (slice(1, -1),)
    elif dim == 2:
        return (slice(1, -1), slice(1, -1))
    else:
        return (slice(1, -1), slice(1, -1), slice(1, -1))
