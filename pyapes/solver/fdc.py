#!/usr/bin/env python3
"""Finite Difference for the current field `FDC`. Similar to Openfoam's `FVC` class.
Each discretization method should create `A_coeffs` and `rhs_adj` attributes.
The `A_coeffs` contains `Ap`, `Ac`, and `Am` and each coefficient has a dimension of `mesh.dim x var.dim x mesh.nx`. Be careful! leading dimension is `mesh.dim` and not `var.dim`.
"""
import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from pyapes.geometry.basis import n2d_coord
from pyapes.solver.tools import default_A_ops
from pyapes.solver.types import DiscretizerConfigType
from pyapes.solver.types import DivConfigType
from pyapes.variables import Field
from pyapes.variables.bcs import BC
from pyapes.variables.container import Hess, Jac

from pymytools.indices import tensor_idx


@dataclass
class Discretizer(ABC):
    """Collection of the operators for explicit finite difference discretization.
    Currently, all operators are meaning in `var[:][inner_slicer(mesh.dim)]` region.
    Therefore, to use in the `FDM` solver, the boundary conditions `var` should be applied before/during the `linalg` process.
    """

    A_coeffs: list[list[Tensor]] | None = None
    """Tuple of A operation matrix coefficients."""
    rhs_adj: Tensor | None = None
    """RHS adjustment tensor."""
    _op_type: str = "Discretizer"
    _config: DiscretizerConfigType | None = None

    @property
    def op_type(self) -> str:
        return self._op_type

    @property
    def config(self) -> DiscretizerConfigType | None:
        return self._config

    @staticmethod
    @abstractmethod
    def build_A_coeffs(
        *args: Field | Tensor | float, config: DiscretizerConfigType | None = None
    ) -> list[list[Tensor]]:
        """Build the operation matrix coefficients to be used for the discretization.
        `var: Field` is required due to the boundary conditions. Should always return three tensors in `Ap`, `Ac`, and `Am` order.
        """
        ...

    @staticmethod
    @abstractmethod
    def adjust_rhs(
        *args: Field | Tensor | float, config: DiscretizerConfigType | None = None
    ) -> Tensor:
        """Return a tensor that is used to adjust `rhs` of the PDE."""
        ...

    def apply(self, A_coeffs: list[list[Tensor]], var: Field) -> Tensor:
        """Apply the discretization to the input `Field` variable."""

        assert A_coeffs is not None, "FDC: A_A_coeffs is not defined!"

        # Grad operator returns Jacobian, but Laplacian, Div, and Ddt return scalar (sum over j-index)
        if self.op_type == "Grad":
            dis_var_dim = []
            for idx in range(var.dim):
                grad_d = []
                for dim in range(var.mesh.dim):
                    grad_d.append(_A_coeff_var_sum(A_coeffs, var, idx, dim))
                dis_var_dim.append(torch.stack(grad_d))
            discretized = torch.stack(dis_var_dim)
        else:
            discretized = torch.zeros_like(var())

            for idx in range(var.dim):
                for dim in range(var.mesh.dim):
                    discretized[idx] += _A_coeff_var_sum(A_coeffs, var, idx, dim)

        return discretized

    def reset(self) -> None:
        """Resetting all the attributes to `None`."""

        self.A_coeffs = None
        self.rhs_adj = None

    def set_config(self, config: DiscretizerConfigType) -> None:
        """Set the configuration for the discretization."""

        self._config = config

    def __call__(
        self, *args: Field | Tensor | float, edge: bool = False
    ) -> Tensor | list[Tensor]:
        """By calling the class with the input `Field` variable, the discretization is conducted."""

        if len(args) == 1:
            assert isinstance(args[0], Field), "FDC: only `Field` is allowed for var!"
            return self.__call_one_var(args[0], edge)
        else:
            assert isinstance(
                args[0], Field | Tensor | float
            ), "FDC: for var_j, Field, Tensor or float is allowed!"
            assert isinstance(args[1], Field), "FDC: only `Field` is allowed for var_i!"

            return self.__call_two_vars(args[0], args[1], edge)

    def __call_one_var(self, var: Field, edge: bool) -> Tensor | list[Tensor]:
        """Return of `__call__` method for the operators that only require one variable."""

        if self.A_coeffs is None:
            self.A_coeffs = self.build_A_coeffs(var)

        if self.rhs_adj is None:
            self.rhs_adj = self.adjust_rhs(var)

        if edge:
            discretized = self.apply(self.A_coeffs, var)
            for dim in range(var.dim):
                _treat_edge(discretized, var, self.op_type, dim)
            return discretized
        else:
            return self.apply(self.A_coeffs, var)

    def __call_two_vars(
        self, var_j: Field | Tensor | float, var_i: Field, edge: bool
    ) -> Tensor:
        """Return of `__call__` method for the operators that require two variables."""

        if self.A_coeffs is None:
            self.A_coeffs = self.build_A_coeffs(var_j, var_i, config=self.config)

        if self.rhs_adj is None:
            self.rhs_adj = self.adjust_rhs(var_j, var_i, config=self.config)

        if edge:
            discretized = self.apply(self.A_coeffs, var_i)
            for dim in range(var_i.dim):
                _treat_edge(discretized, var_i, self.op_type, dim)
            return discretized
        else:
            return self.apply(self.A_coeffs, var_i)


def _A_coeff_var_sum(
    A_coeffs: list[list[Tensor]], var: Field, idx: int, dim: int
) -> Tensor:
    """Sum the coefficients and the variable.
    Here, `len(A_coeffs) = 5` to implement `quick` scheme for `div` operator in the future.
    """

    assert (
        len(A_coeffs) == 5
    ), "FDC: the total number of coefficient tensor should be 5!"

    summed = torch.zeros_like(var()[idx])

    for i, c in enumerate(A_coeffs):
        summed += c[dim][idx] * torch.roll(var()[idx], -2 + i, dim)

    return summed


def _treat_edge(
    discretized: Tensor | list[Tensor], var: Field, ops: str, dim: int
) -> None:
    """Treat edge of discretized variable using the forward/backward difference.
    Here the edge means the domain (mesh) boundary.

    Note:
        - Using slicers is inspired from `numpy.gradient` function
    """

    # Slicers
    slicer_1: list[slice | int] = [slice(None) for _ in range(var.mesh.dim)]
    slicer_2: list[slice | int] = [slice(None) for _ in range(var.mesh.dim)]
    slicer_3: list[slice | int] = [slice(None) for _ in range(var.mesh.dim)]
    slicer_4: list[slice | int] = [slice(None) for _ in range(var.mesh.dim)]

    if ops == "Laplacian":
        # Treat edge with the second order forward/backward difference

        for idx in range(var.mesh.dim):
            slicer_1[idx] = 0
            slicer_2[idx] = 1
            slicer_3[idx] = 2
            slicer_4[idx] = 3

            bc_val = var()[dim][slicer_1]
            bc_val_p = var()[dim][slicer_2]
            bc_val_pp = var()[dim][slicer_3]
            bc_val_ppp = var()[dim][slicer_4]

            discretized[dim][slicer_1] = (
                2.0 * bc_val - 5.0 * bc_val_p + 4.0 * bc_val_pp - bc_val_ppp
            ) / (var.mesh.dx[idx] ** 2)

            slicer_1[idx] = -1
            slicer_2[idx] = -2
            slicer_3[idx] = -3
            slicer_4[idx] = -4

            bc_val = var()[dim][slicer_1]
            bc_val_p = var()[dim][slicer_2]
            bc_val_pp = var()[dim][slicer_3]
            bc_val_ppp = var()[dim][slicer_4]

            discretized[dim][slicer_1] = (
                2.0 * bc_val - 5.0 * bc_val_p + 4.0 * bc_val_pp - bc_val_ppp
            ) / (var.mesh.dx[idx] ** 2)

            slicer_1[idx] = slice(None)
            slicer_2[idx] = slice(None)
            slicer_3[idx] = slice(None)
            slicer_4[idx] = slice(None)

    elif ops == "Grad":
        for idx in range(var.mesh.dim):
            slicer_1[idx] = 0
            slicer_2[idx] = 1
            slicer_3[idx] = 2

            bc_val = var()[dim][slicer_1]
            bc_val_p = var()[dim][slicer_2]
            bc_val_pp = var()[dim][slicer_3]

            discretized[dim][idx][slicer_1] = -(
                3 / 2 * bc_val - 2.0 * bc_val_p + 1 / 2 * bc_val_pp
            ) / (var.mesh.dx[idx])

            slicer_1[idx] = -1
            slicer_2[idx] = -2
            slicer_3[idx] = -3

            bc_val = var()[dim][slicer_1]
            bc_val_p = var()[dim][slicer_2]
            bc_val_pp = var()[dim][slicer_3]

            discretized[dim][idx][slicer_1] = (
                3 / 2 * bc_val - 2.0 * bc_val_p + 1 / 2 * bc_val_pp
            ) / (var.mesh.dx[idx])

            slicer_1[idx] = slice(None)
            slicer_2[idx] = slice(None)
            slicer_3[idx] = slice(None)

    elif ops == "Div":
        warnings.warn(
            "FDC: edge treatment of Div is supported! Div assumes user already specified the boundary in the domain."
        )
    else:
        raise RuntimeError(f"FDC: edge treatment of {ops=} is not supported!")

    pass


class Laplacian(Discretizer):
    """Laplacian discretizer."""

    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(var: Field) -> list[list[Tensor]]:
        App, Ap, Ac, Am, Amm = default_A_ops(var, __class__.__name__)

        dx = var.dx
        # Treat boundaries
        for i in range(var.dim):
            for j in range(var.mesh.dim):
                if var.bcs is None:
                    # Do nothing
                    continue

                # Treat BC
                for bc in var.bcs:
                    # If discretization direction is not the same as the BC surface normal direction, do nothing
                    if bc.bc_n_vec[j] == 0:
                        continue

                    if bc.bc_type == "neumann" or bc.bc_type == "symmetry":
                        # Treatment for the cylindrical coordinate
                        dr = var.mesh.dx[j] if j == 0 else 0.0
                        r = var.mesh.grid[j][bc.bc_mask_prev]
                        alpha = (
                            torch.nan_to_num(
                                2 / 3 * dr / r, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            if var.mesh.coord_sys == "rz"
                            else torch.zeros_like(r)
                        )

                        if bc.bc_n_dir < 0:
                            # At lower side
                            Ap[j][i][bc.bc_mask_prev] = 2 / 3 + alpha
                            Ac[j][i][bc.bc_mask_prev] = -(2 / 3 + alpha)
                            Am[j][i][bc.bc_mask_prev] = 0.0
                        else:
                            # At upper side
                            Ap[j][i][bc.bc_mask_prev] = 0.0
                            Ac[j][i][bc.bc_mask_prev] = -(2 / 3 + alpha)
                            Am[j][i][bc.bc_mask_prev] = 2 / 3 + alpha
                    else:
                        # Do nothing
                        pass

                Ap[j][i] /= dx[j] ** 2
                Ac[j][i] /= dx[j] ** 2
                Am[j][i] /= dx[j] ** 2

        return [App, Ap, Ac, Am, Amm]

    @staticmethod
    def adjust_rhs(var: Field) -> Tensor:
        rhs_adj = torch.zeros_like(var())
        dx = var.dx

        # Treat boundaries
        for i in range(var.dim):
            if var.bcs is None:
                # Do nothing
                continue

            for j in range(var.mesh.dim):
                for bc in var.bcs:
                    if bc.bc_type == "neumann":
                        # Treatment for the cylindrical coordinate
                        dr = var.mesh.dx[j] if j == 0 else 0.0
                        r = var.mesh.grid[j][bc.bc_mask_prev]
                        alpha = (
                            torch.nan_to_num(
                                1 / 3 * dr / r, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            if var.mesh.coord_sys == "rz"
                            else torch.zeros_like(r)
                        )

                        at_bc = _return_bc_val(bc, var, i)
                        rhs_adj[i][bc.bc_mask_prev] += (
                            (2 / 3 - alpha) * (at_bc * bc.bc_n_vec[j]) / dx[j]
                        )
                    else:
                        # Do nothing
                        pass

        return rhs_adj


class Grad(Discretizer):
    """Gradient operator.
    Once the discretization is conducted, returned value is a `2 + len(mesh.nx)` dimensional tensor with the shape of `(var.dim, mesh.dim, *mesh.nx)`

    Example:

    >>> mesh = Mesh(Box[0:1, 0:1], None, [10, 10]) # 2D mesh with 10x10 cells
    >>> var = Field("test_field", 1, mesh, ...) # scalar field
    >>> fdm = FDM()
    >>> grad = fdm.grad(var)
    >>> grad.shape
    torch.Size([1, 2, 10, 10])

    """

    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(var: Field) -> list[list[Tensor]]:
        r"""Build the coefficients for the discretization of the gradient operator using the second-order central finite difference method.

        ..math::
            \nabla \Phi = \frac{\Phi^{i+1} - \Phi^{i-1}}{2 \Delta x}
        """
        App, Ap, Ac, Am, Amm = default_A_ops(var, __class__.__name__)

        if var.bcs is not None:
            for i in range(var.dim):
                _grad_central_adjust(var, [Ap, Ac, Am], i)

        return [App, Ap, Ac, Am, Amm]

    @staticmethod
    def adjust_rhs(var: Field) -> Tensor:
        rhs_adj = torch.zeros_like(var())

        if var.bcs is not None:
            for i in range(var.dim):
                _grad_rhs_adjust(var, rhs_adj, i)

        return rhs_adj


def _grad_rhs_adjust(
    var: Field, rhs_adj: Tensor, dim: int, gamma: tuple[Tensor, ...] | None = None
) -> None:
    """Adjust the RHS for the gradient operator. This function is seperated from the class to be reused in the `Div` operator."""

    if gamma is None:
        gamma_min = torch.ones_like(var())
        gamma_max = torch.ones_like(var())
    else:
        if len(gamma) == 1:
            gamma_min = 2.0 * gamma[0]
            gamma_max = 2.0 * gamma[0]
        else:
            gamma_min = 2.0 * gamma[0]
            gamma_max = 2.0 * gamma[1]

    for j in range(var.mesh.dim):
        for bc in var.bcs:
            if bc.bc_type == "neumann":
                at_bc = _return_bc_val(bc, var, dim)

                if bc.bc_n_dir < 0:
                    rhs_adj[dim][bc.bc_mask_prev] -= (
                        (1 / 3)
                        * (at_bc * bc.bc_n_vec[j])
                        * gamma_max[dim][bc.bc_mask_prev]
                    )
                else:
                    rhs_adj[dim][bc.bc_mask_prev] -= (
                        (1 / 3)
                        * (at_bc * bc.bc_n_vec[j])
                        * gamma_min[dim][bc.bc_mask_prev]
                    )
            else:
                # Dirichlet and Symmetry BC: Do nothing
                pass


def _grad_central_adjust(
    var: Field,
    A_ops: list[list[Tensor]],
    dim: int,
    gamma: tuple[Tensor, ...] | None = None,
) -> None:
    """Adjust gradient's A_ops to accommodate boundary conditions.

    Args:
        var (Field): input variable to be discretized
        A_ops (tuple[list[Tensor], ...]): tuple of lists of tensors containing the coefficients of the discretization. `len(A_ops) == 3` since we need `Ap`, `Ac`, and `Am` coefficients.
        dim (int): variable dimension. It should be in the range of `var.dim`. Defaults to 0.
        it is not the dimension of the mesh!
    """

    if gamma is None:
        gamma_min = torch.ones_like(var())
        gamma_max = torch.ones_like(var())
    else:
        if len(gamma) == 1:
            gamma_min = gamma[0]
            gamma_max = gamma[0]
        else:
            gamma_min = gamma[0]
            gamma_max = gamma[1]

    Ap, Ac, Am = A_ops

    dx = var.dx

    # Treat boundaries
    for j in range(var.mesh.dim):
        # Treat BC
        for bc in var.bcs:
            # If discretization direction is not the same as the BC surface normal direction, do nothing
            if bc.bc_n_vec[j] == 0:
                continue

            if bc.bc_type == "neumann" or bc.bc_type == "symmetry":
                gmx_at_mask = gamma_max[dim][bc.bc_mask_prev]
                gmn_at_mask = gamma_min[dim][bc.bc_mask_prev]

                if bc.bc_n_dir < 0:
                    # At lower side
                    Ap[j][dim][bc.bc_mask_prev] += 1 / 3 * gmx_at_mask
                    Ac[j][dim][bc.bc_mask_prev] -= 1 / 3 * gmn_at_mask
                    Am[j][dim][bc.bc_mask_prev] = 0.0
                else:
                    # At upper side
                    Ap[j][dim][bc.bc_mask_prev] = 0.0
                    Ac[j][dim][bc.bc_mask_prev] += 1 / 3 * gmn_at_mask
                    Am[j][dim][bc.bc_mask_prev] -= 1 / 3 * gmx_at_mask
            elif bc.bc_type == "periodic":
                if bc.bc_n_dir < 0:
                    # At lower side
                    Am[j][dim][bc.bc_mask_prev] = 0.0
                else:
                    # At upper side
                    Ap[j][dim][bc.bc_mask_prev] = 0.0
            else:
                # Dirichlet BC: Do nothing
                pass

        Ap[j][dim] /= 2.0 * dx[j]
        Ac[j][dim] /= 2.0 * dx[j]
        Am[j][dim] /= 2.0 * dx[j]


class Div(Discretizer):
    """Divergence operator.
    It supports `central` and `upwind` discretization methods.

    FUTURE: quick scheme
    """

    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(
        var_j: Field | float | Tensor,
        var_i: Field,
        config: DiscretizerConfigType,
    ) -> list[list[Tensor]]:
        r"""Build the coefficients for the discretization of the gradient operator using the second-order central finite difference method. `i` and `j` indicates the Einstein summation convention. Here, `j` comes first to be consistent with the equation:

        ..math::
            \vec{u}_{j} \frac{\partial \Phi}{\partial x_j} = \frac{\Phi^{i+1} - \Phi^{i-1}}{2 \Delta x} + ...

        Args:
            var_j (Field | float | Tensor): advection term
            var_i (Field): input variable to be discretized
            config (dict[str, str]): configuration dictionary. It should contain the following keys: `limiter`.
        """

        adv = _div_var_j_to_tensor(var_j, var_i)

        assert "div" in config, "FDC Div: config should contain 'div' key."

        limiter = _check_limiter(config["div"])

        App, Ap, Ac, Am, Amm = default_A_ops(var_i, __class__.__name__)

        if limiter == "none":
            Ap, Ac, Am = _adv_central(adv, var_i, [Ap, Ac, Am])
        elif limiter == "upwind":
            Ap, Ac, Am = _adv_upwind(adv, var_i)
        elif limiter == "quick":
            raise NotImplementedError("FDC Div: quick scheme is not implemented yet.")
        else:
            raise RuntimeError(f"FDC Div: {limiter=} is an unknown limiter type.")

        return [App, Ap, Ac, Am, Amm]

    @staticmethod
    def adjust_rhs(
        var_j: Field | Tensor | float, var_i: Field, config: DiscretizerConfigType
    ) -> Tensor:
        rhs_adj = torch.zeros_like(var_i())

        if var_i.bcs is not None:
            adv = _div_var_j_to_tensor(var_j, var_i)

            assert "div" in config, "FDC Div: config should contain 'div' key."

            limiter = _check_limiter(config["div"])

            if limiter == "none":
                for i in range(var_i.dim):
                    _grad_rhs_adjust(var_i, rhs_adj, i, (adv,))
            elif limiter == "upwind":
                gamma_min, gamma_max = _gamma_from_adv(adv, var_i)
                for i in range(var_i.dim):
                    _grad_rhs_adjust(var_i, rhs_adj, i, (gamma_min, gamma_max))
            elif limiter == "quick":
                raise NotImplementedError(
                    "FDC Div: quick scheme is not implemented yet."
                )
            else:
                raise RuntimeError(f"FDC Div: {limiter=} is an unknown limiter type.")

        return rhs_adj


def _check_limiter(config: DivConfigType | None) -> str:
    """Check the limiter type."""
    if config is not None and "limiter" in config:
        return config["limiter"].lower()  # make sure it is lower case
    else:
        warnings.warn(
            "FDM: no limiter is specified. Use `none` (central difference) as a default."
        )
        return "none"


def _adv_central(
    adv: Tensor, var: Field, A_ops: list[list[Tensor]]
) -> list[list[Tensor]]:
    """Discretization of the advection tern using central difference.

    Args:
        adv (Tensor): Advection term, i.e., `var_j`.
        var (Field): variable to be discretized. i.e., `var_i`.
        A_ops (tuple[list[Tensor], ...]): Discretization coefficients.
    """

    # Leading dimension is the dimension of the mesh
    # The following dimension is the dimension of the variable
    # A_[mesh.dim][var.dim]
    Ap, Ac, Am = A_ops

    for i in range(var.dim):
        for j in range(var.mesh.dim):
            Ap[j][i] *= adv[i]
            Ac[j][i] *= adv[i]
            Am[j][i] *= adv[i]

        _grad_central_adjust(var, [Ap, Ac, Am], i, (adv,))

    return [Ap, Ac, Am]


def _adv_upwind(adv: Tensor, var: Field) -> list[list[Tensor]]:
    """Upwind discretization of the advection term."""

    gamma_min, gamma_max = _gamma_from_adv(adv, var)

    # Here 2.0 is multiplied to achieve consistent implementation of _grad_central_adjust (all coeffs are divided by 2.0 * dx there)
    Ap = [2.0 * gamma_min for _ in range(var.mesh.dim)]
    Ac = [2.0 * gamma_max - gamma_min for _ in range(var.mesh.dim)]
    Am = [-2.0 * gamma_max for _ in range(var.mesh.dim)]

    for i in range(var.dim):
        _grad_central_adjust(var, [Ap, Ac, Am], i, (gamma_min, gamma_max))

    return [Ap, Ac, Am]


def _div_var_j_to_tensor(var_j: Field | Tensor | float, var_i: Field) -> Tensor:
    """In `Div` operator, convert `var_j` to a `Tensor`. Also check the shape of `var_j` so that is has the same shape of target variable `var_i`."""

    if isinstance(var_j, float):
        adv = torch.ones_like(var_i()) * var_j
    elif isinstance(var_j, Tensor):
        adv = var_j
        # Shape check
        assert adv.shape == var_i().shape, "FDC Div: adv shape must match var_i shape"
    else:
        adv = var_j()

    return adv


def _gamma_from_adv(adv: Tensor, var: Field) -> tuple[Tensor, Tensor]:
    zeros = torch.zeros_like(var())
    gamma_min = torch.min(adv, zeros)
    gamma_max = torch.max(adv, zeros)

    return gamma_min, gamma_max


def _return_bc_val(bc: BC, var: Field, dim: int) -> Tensor | float:
    """Return boundary values."""

    if callable(bc.bc_val):
        at_bc = bc.bc_val(var.mesh.grid, bc.bc_mask, var(), bc.bc_n_vec)
    elif isinstance(bc.bc_val, list):
        at_bc = bc.bc_val[dim]
    elif isinstance(bc.bc_val, float | int):
        at_bc = bc.bc_val
    elif bc.bc_val is None:
        at_bc = 0.0
    else:
        raise ValueError(f"Unknown boundary condition value: {bc.bc_val}")

    return at_bc


class DiffFlux:
    """Object to be used in the tensor diffussion term."""

    @staticmethod
    def __call__(diff: Hess, var: Field) -> Field:
        r"""Compute the diffusive flux without boundary treatment (just forward-backward difference)

        .. math::
            D_ij \frac{\partial \Phi}{\partial x_j}

        Therefore, it returns a vector field.

        Args:
            diff (Hess): Diffusion tensor
            var (Field): Scalar input field
        """

        # FIXME!
        jac = ScalarOP.jac(var)
        flux = Field("DiffFlux", len(jac), var.mesh, None)

        n2d = n2d_coord(var.mesh.coord_sys)

        for i in range(var.mesh.dim):
            diff_flux = torch.zeros_like(var()[0])
            for j in range(var.mesh.dim):
                j_key = n2d[j]
                h_key = n2d[i] + n2d[j]

                if n2d[i] == "r":
                    d_coeff = var.mesh.grid[0] * diff[h_key]
                else:
                    d_coeff = diff[h_key]

                diff_flux += d_coeff * jac[j_key]

            flux.set_var_tensor(diff_flux, i)

        return flux


class FDC:
    """Collection of Finite Difference discretization. The operation is explicit, therefore, all methods return a tensor."""

    div: Div = Div()
    """Divergence operator: `div(var_j, var_i)`."""
    laplacian: Laplacian = Laplacian()
    """Laplacian operator: `laplacian(coeffs, var)`."""
    grad: Grad = Grad()
    """Gradient operator: `grad(var)`."""
    diffFlux: DiffFlux = DiffFlux()

    def __init__(self, config: DiscretizerConfigType | None = None):
        """Init FDC class."""

        self.config = config

    def update_config(self, scheme: str, target: str, val: str):
        """Update config values."""

        if self.config is not None:
            self.config[scheme][target] = val
        else:
            self.config = {scheme: {target: val}}


class ScalarOP:
    """Manipulation of a scalar field (scalar operations)

    Note:
        - `jac` and `hess` operations both use the `torch.gradient` function with edge order of 2.
    """

    @staticmethod
    def jac(var: Field) -> Jac:
        assert var().shape[0] == 1, "Scalar: var must be a scalar field."

        data_jac: dict[str, Tensor] = {}

        n2d = n2d_coord(var.mesh.coord_sys)

        jac = FDC.grad(var, edge=True)[0]

        for i, j in enumerate(jac):
            data_jac[n2d[i]] = j

        FDC.grad.reset()

        return Jac(**data_jac)

    @staticmethod
    def hess(var: Field) -> Hess:
        indices = tensor_idx(var.mesh.dim)

        data_hess: dict[str, Tensor] = {}

        hess: list[Tensor] = []

        n2d = n2d_coord(var.mesh.coord_sys)

        jac = FDC.grad(var, edge=True)[0]

        jac_f = var.copy()

        hess = [FDC.grad(jac_f.set_var_tensor(j), edge=True)[0] for j in jac]

        for i, hi in enumerate(hess):
            for j, h in enumerate(hi):
                if (i, j) in indices:
                    data_hess[n2d[i] + n2d[j]] = h

        FDC.grad.reset()
        return Hess(**data_hess)
