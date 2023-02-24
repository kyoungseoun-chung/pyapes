#!/usr/bin/env python3
"""Finite Difference for the current field `FDC`. Similar to Openfoam's `FVC` class.
Each discretization method should create `A_coeffs` and `rhs_adj` attributes.
The `A_coeffs` contains `Ap`, `Ac`, and `Am` and each coefficient has a dimension of `mesh.dim x var.dim x mesh.nx`. Be careful! leading dimension is `mesh.dim` and not `var.dim`.
"""
import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from pyapes.core.solver.tools import default_A_ops
from pyapes.core.variables import Field
from pyapes.core.variables.bcs import BC


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
    _config: dict[str, str] | None = None

    @property
    def op_type(self) -> str:
        return self._op_type

    @property
    def config(self) -> dict[str, str] | None:
        return self._config

    @staticmethod
    @abstractmethod
    def build_A_coeffs(
        *args: Field | Tensor | float, config: dict[str, str] | None = None
    ) -> list[list[Tensor]]:
        """Build the operation matrix coefficients to be used for the discretization.
        `var: Field` is required due to the boundary conditions. Should always return three tensors in `Ap`, `Ac`, and `Am` order.
        """
        ...

    @staticmethod
    @abstractmethod
    def adjust_rhs(
        *args: Field | Tensor | float, config: dict[str, str] | None = None
    ) -> Tensor:
        """Return a tensor that is used to adjust `rhs` of the PDE."""
        ...

    def apply(self, coeffs: list[list[Tensor]], var: Field) -> Tensor:
        """Apply the discretization to the input `Field` variable."""

        assert coeffs is not None, "FDC: A_coeffs is not defined!"

        # Grad operator returns Jacobian but Laplacian, Div, and Ddt return scalar (sum over j-index)
        if self.op_type == "Grad":
            dis_var_dim = []
            for idx in range(var.dim):
                grad_d = []
                for dim in range(var.mesh.dim):
                    grad_d.append(_coeff_var_sum(coeffs, var, idx, dim))
                dis_var_dim.append(torch.stack(grad_d))
            discretized = torch.stack(dis_var_dim)
        else:
            discretized = torch.zeros_like(var())

            for idx in range(var.dim):
                for dim in range(var.mesh.dim):
                    discretized[idx] += _coeff_var_sum(coeffs, var, idx, dim)

        return discretized

    def reset(self) -> None:
        """Resetting all the attributes to `None`."""

        self.A_coeffs = None
        self.rhs_adj = None

    def set_config(self, config: dict[str, str]) -> None:
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
        """Return of `__call__` method for the operators that only require one variable.

        * This is private method.
        """

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
        """Return of `__call__` method for the operators that require two variables.

        * This is private method.
        """

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


def _coeff_var_sum(
    coeffs: list[list[Tensor]], var: Field, idx: int, dim: int
) -> Tensor:
    """Sum the coefficients and the variable.
    Here, `len(coeffs) = 5` to implement `quick` scheme for `div` operator in the future.
    """

    assert len(coeffs) == 5, "FDC: the total number of coefficient tensor should be 5!"

    summed = torch.zeros_like(var()[idx])

    for i, c in enumerate(coeffs):
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

    elif ops == "Div":
        raise NotImplementedError(
            f"FDC: edge treatment of {ops=} is not implemented yet!"
        )
    else:
        raise RuntimeError(f"FDC: edge treatment of {ops=} is not supported!")


class Laplacian(Discretizer):
    """Laplacian discretizer."""

    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(var: Field) -> list[list[Tensor]]:

        App, Ap, Ac, Am, Amm = default_A_ops(var, 2)

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
                        if bc.bc_n_dir < 0:
                            # At lower side
                            Ap[j][i][bc.bc_mask_prev] = 2 / 3
                            Ac[j][i][bc.bc_mask_prev] = -2 / 3
                            Am[j][i][bc.bc_mask_prev] = 0.0
                        else:
                            # At upper side
                            Ap[j][i][bc.bc_mask_prev] = 0.0
                            Ac[j][i][bc.bc_mask_prev] = -2 / 3
                            Am[j][i][bc.bc_mask_prev] = 2 / 3
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
                        at_bc = _return_bc_val(bc, var, i)
                        rhs_adj[i][bc.bc_mask_prev] += (
                            (2 / 3) * (at_bc * bc.bc_n_vec[j]) / dx[j]
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
        App, Ap, Ac, Am, Amm = default_A_ops(var, 1)

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
    var: Field, rhs_adj: Tensor, dim: int, adv: Tensor | None = None
) -> None:
    """Adjust the RHS for the gradient operator. This function is seperated from the class to be reused in the `Div` operator."""

    if adv is None:
        adv = torch.ones_like(var())

    dx = var.dx

    for j in range(var.mesh.dim):

        for bc in var.bcs:
            if bc.bc_type == "neumann":
                at_bc = _return_bc_val(bc, var, dim)
                rhs_adj[dim][bc.bc_mask_prev] += (
                    (1 / 3) * (at_bc * bc.bc_n_vec[j]) * adv[dim][bc.bc_mask_prev]
                )
            elif bc.bc_type == "periodic":

                rhs_adj[dim][bc.bc_mask_prev] += (
                    var()[dim][bc.bc_mask_prev]
                    / (2.0 * dx[j])
                    * float(bc.bc_n_dir)
                    * adv[dim][bc.bc_mask_prev]
                )
            else:
                # Dirichlet and Symmetry BC: Do nothing
                pass


def _grad_central_adjust(
    var: Field, A_ops: list[list[Tensor]], dim: int, adv: Tensor | None = None
) -> None:
    """Adjust gradient's A_ops to accommodate boundary conditions.

    Args:
        var (Field): input variable to be discretized
        A_ops (tuple[list[Tensor], ...]): tuple of lists of tensors containing the coefficients of the discretization. `len(A_ops) == 3` since we need `Ap`, `Ac`, and `Am` coefficients.
        dim (int): variable dimension. It should be in the range of `var.dim`. Defaults to 0.
        it is not the dimension of the mesh!
    """

    if adv is None:
        adv = torch.ones_like(var())

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

                adv_at_mask_p = adv[dim][bc.bc_mask_prev]

                if bc.bc_n_dir < 0:
                    # At lower side
                    Ap[j][dim][bc.bc_mask_prev] = 4 / 3 * adv_at_mask_p
                    Ac[j][dim][bc.bc_mask_prev] = -4 / 3 * adv_at_mask_p
                    Am[j][dim][bc.bc_mask_prev] = 0.0
                else:
                    # At upper side
                    Ap[j][dim][bc.bc_mask_prev] = 0.0
                    Ac[j][dim][bc.bc_mask_prev] = -4 / 3 * adv_at_mask_p
                    Am[j][dim][bc.bc_mask_prev] = 4 / 3 * adv_at_mask_p
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

    WIP
    FUTURE: quick scheme
    """

    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(
        var_j: Field | float | Tensor, var_i: Field, config: dict[str, str]
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

        limiter = _check_limiter(config)

        App, Ap, Ac, Am, Amm = default_A_ops(var_i, 1)

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
    def adjust_rhs(var_j: Field, var_i: Field, config: dict[str, str]) -> Tensor:

        rhs_adj = torch.zeros_like(var_i())

        if var_i.bcs is not None:

            adv = _div_var_j_to_tensor(var_j, var_i)
            limiter = _check_limiter(config)

            if limiter == "none":
                for i in range(var_i.dim):
                    _grad_rhs_adjust(var_i, rhs_adj, i, adv)
            elif limiter == "upwind":
                pass
            elif limiter == "quick":
                raise NotImplementedError(
                    "FDC Div: quick scheme is not implemented yet."
                )
            else:
                raise RuntimeError(f"FDC Div: {limiter=} is an unknown limiter type.")

        return rhs_adj


def _check_limiter(config: dict[str, str] | None) -> str:
    """Check the limiter type."""
    if config is not None and "limiter" in config:
        return config["limiter"].lower()  # make sure it is lower case
    else:
        warnings.warn(
            "FDM: no limiter is specified. Use `none` (central difference) as a default."
        )
        return "none"


# NOTE: wrong! need var.dim since adv = var_j
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
    _, Ap, Ac, Am, _ = A_ops

    for i in range(var.dim):

        for j in range(var.mesh.dim):

            Ap[j][i] *= adv[i]
            Ac[j][i] *= adv[i]
            Am[j][i] *= adv[i]

        _grad_central_adjust(var, [Ap, Ac, Am], i)

    return [Ap, Ac, Am]


def _adv_upwind(adv: Tensor, var: Field) -> list[list[Tensor]]:
    """Upwind discretization of the advection term."""

    zeros = torch.zeros_like(var())
    gamma_min = torch.min(adv, zeros)
    gamma_max = torch.max(adv, zeros)

    Ap = [gamma_min for _ in range(var.mesh.dim)]
    Ac = [gamma_min + gamma_max for _ in range(var.mesh.dim)]
    Am = [-gamma_max for _ in range(var.mesh.dim)]

    for i in range(var.dim):
        _div_upwind_adjust(var, [Ap, Ac, Am], gamma_min, gamma_max, i)

    return [Ap, Ac, Am]


# NOTE: maybe we can merge this with _grad_central_adjust?
def _div_upwind_adjust(
    var: Field,
    A_ops: list[list[Tensor]],
    gamma_min: Tensor,
    gamma_max: Tensor,
    dim: int,
) -> None:
    """Adjust `Div`'s upwind A_ops to accommodate boundary conditions. Similar to `_grad_upwind_adjust`.

    Args:
        var (Field): input variable to be discretized
        A_ops (tuple[list[Tensor], ...]): tuple of lists of tensors containing the coefficients of the discretization. `len(A_ops) == 3` since we need `Ap`, `Ac`, and `Am` coefficients.
        dim (int): variable dimension. It should be in the range of `var.dim`. Defaults to 0.
        it is not the dimension of the mesh!
    """

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

                if bc.bc_n_dir < 0:

                    adv_p = (
                        1 / 3 * gamma_max[j][dim][bc.bc_mask_prev]
                        + gamma_min[j][dim][bc.bc_mask_prev]
                    )
                    adv_c = (
                        -1 / 3 * gamma_max[j][dim][bc.bc_mask_prev]
                        - gamma_min[j][dim][bc.bc_mask_prev]
                    )

                    # At lower side
                    Ap[j][dim][bc.bc_mask_prev] = adv_p
                    Ac[j][dim][bc.bc_mask_prev] = adv_c
                    Am[j][dim][bc.bc_mask_prev] = 0.0
                else:
                    adv_c = (
                        gamma_max[j][dim][bc.bc_mask_prev]
                        + 1 / 3 * gamma_min[j][dim][bc.bc_mask_prev]
                    )
                    adv_m = (
                        -gamma_max[j][dim][bc.bc_mask_prev]
                        - 1 / 3 * gamma_min[j][dim][bc.bc_mask_prev]
                    )
                    # At upper side
                    Ap[j][dim][bc.bc_mask_prev] = 0.0
                    Ac[j][dim][bc.bc_mask_prev] = adv_c
                    Am[j][dim][bc.bc_mask_prev] = adv_m
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


class Ddt(Discretizer):
    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(var: Field) -> tuple[Tensor, Tensor, Tensor]:
        ...

    @staticmethod
    def adjust_rhs(var: Field) -> Tensor:
        ...


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


class FDC:
    """Collection of Finite Difference discretization. The operation is explicit, therefore, all methods return a tensor."""

    config: Optional[dict[str, dict[str, str]]] = None
    """Configuration for the discretization."""
    div: Div = Div()
    """Divergence operator: `div(var_j, var_i)`."""
    laplacian: Laplacian = Laplacian()
    """Laplacian operator: `laplacian(coeffs, var)`."""
    grad: Grad = Grad()
    """Gradient operator: `grad(var)`."""
    ddt: Ddt = Ddt()
    """Time discretization: `ddt(var)`. It will only adjust RHS of the PDE."""

    def update_config(self, scheme: str, target: str, val: str):
        """Update config values."""

        if self.config is not None:
            self.config[scheme][target] = val
        else:
            self.config = {scheme: {target: val}}


# NOTE: Legacy code. Will be removed in the future.
# @dataclass
# class FDC_old:
#     """Collection of the operators for explicit finite difference discretization."""

#     def ddt(self, dt: float, var: Field) -> Tensor:
#         """Time derivative of a given field.

#         Note:
#             - If `var` does not have old `VARo`, attribute, current `VAR` will be treated as `VARo`.
#         """

#         ddt = []

#         for i in range(var.dim):
#             try:
#                 var_o = var.VARo[i]
#             except AttributeError:
#                 # If not saved, use current value (treat for the first iteration)
#                 var_o = var()[i]
#             ddt.append((var()[i] - var_o) / dt)

#         return torch.stack(ddt)

#     def div(self, var_j: Field, var_i: Field) -> Tensor:
#         """Divergence of two fields.
#         Note:
#             - To avoid the checkerboard problem, flux limiter is used. It supports `none`, `upwind` and `quick` limiter. (here, `none` is equivalent to the second order central difference.)
#         """

#         if self.config is not None and "limiter" in self.config["div"]:
#             limiter = self.config["div"]["limiter"]
#         else:
#             warnings.warn("FDM: no limiter is specified. Use `none` as default.")
#             limiter = "none"

#         if var_j.name == var_i.name:
#             # If var_i and var_j are the same field, use the same tensor.
#             var_j.set_var_tensor(var_i().clone())

#         div = []

#         dx = var_j.dx

#         for i in range(var_i.dim):

#             d_val = torch.zeros_like(var_i()[i])

#             for j in range(var_j.dim):

#                 if limiter == "none":
#                     """Central difference scheme."""

#                     pad = create_pad(var_i.mesh.dim)
#                     slicer = inner_slicer(var_i.mesh.dim)

#                     bc_il = var_i.get_bc(f"d-{NUM_TO_DIR[j]}l")
#                     bc_ir = var_i.get_bc(f"d-{NUM_TO_DIR[j]}r")

#                     # m_val = fill_pad(pad(var_i()[i]), j, 1, slicer)
#                     m_val = fill_pad_bc(pad(var_i()[i]), 1, slicer, [bc_il, bc_ir], j)

#                     d_val += (
#                         var_j()[j]
#                         * (
#                             (torch.roll(m_val, -1, j) - torch.roll(m_val, 1, j))
#                             / (2 * dx[j])
#                         )[slicer]
#                     )

#                 elif limiter == "upwind":

#                     pad = create_pad(var_i.mesh.dim)
#                     slicer = inner_slicer(var_i.mesh.dim)

#                     var_i_pad = fill_pad(pad(var_i()[i]), j, 1, slicer)
#                     var_j_pad = fill_pad(pad(var_j()[j]), j, 1, slicer)

#                     m_val_p = (torch.roll(var_j_pad, -1, j) + var_j_pad) / 2
#                     m_val_m = (torch.roll(var_j_pad, 1, j) + var_j_pad) / 2

#                     f_val_p = (m_val_p + m_val_p.abs()) * var_i_pad / 2 - (
#                         m_val_p - m_val_p.abs()
#                     ) * torch.roll(var_i_pad, -1, j) / 2

#                     f_val_m = (m_val_m + m_val_m.abs()) * torch.roll(
#                         var_i_pad, 1, j
#                     ) / 2 - (m_val_p - m_val_p.abs()) * var_i_pad / 2

#                     d_val += ((f_val_p - f_val_m) / dx[j])[slicer]

#                 elif limiter == "quick":
#                     pad = create_pad(var_i.mesh.dim, 2)
#                     slicer = inner_slicer(var_i.mesh.dim, 2)

#                     pass
#                 else:
#                     raise ValueError("FDM: Unknown limiter.")

#             div.append(d_val)

#         return torch.stack(div)
