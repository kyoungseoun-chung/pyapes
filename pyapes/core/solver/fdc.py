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

from pyapes.core.geometry.basis import NUM_TO_DIR
from pyapes.core.mesh.tools import create_pad
from pyapes.core.mesh.tools import inner_slicer
from pyapes.core.solver.tools import fill_pad
from pyapes.core.solver.tools import fill_pad_bc
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

    @property
    def op_type(self) -> str:
        return self._op_type

    @staticmethod
    @abstractmethod
    def build_A_coeffs(var: Field) -> list[list[Tensor]]:
        """Build the operation matrix coefficients to be used for the discretization.
        `var: Field` is required due to the boundary conditions. Should always return three tensors in `Ap`, `Ac`, and `Am` order.
        """
        ...

    @staticmethod
    @abstractmethod
    def adjust_rhs(var: Field) -> Tensor:
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

    def __call__(self, var: Field, edge: bool = False) -> Tensor | list[Tensor]:
        """By calling the class with the input `Field` variable, the discretization is conducted."""

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

        App = [torch.zeros_like(var()) for _ in range(var.mesh.dim)]
        Ap = [torch.ones_like(var()) for _ in range(var.mesh.dim)]
        Ac = [-2.0 * torch.ones_like(var()) for _ in range(var.mesh.dim)]
        Am = [torch.ones_like(var()) for _ in range(var.mesh.dim)]
        Amm = [torch.zeros_like(var()) for _ in range(var.mesh.dim)]

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
        App = [torch.zeros_like(var()) for _ in range(var.mesh.dim)]
        Ap = [torch.ones_like(var()) for _ in range(var.mesh.dim)]
        Ac = [torch.zeros_like(var()) for _ in range(var.mesh.dim)]
        Am = [-1.0 * torch.ones_like(var()) for _ in range(var.mesh.dim)]
        Amm = [torch.zeros_like(var()) for _ in range(var.mesh.dim)]

        for i in range(var.dim):

            _grad_central_adjust(var, (Ap, Ac, Am), i)

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
                        rhs_adj[i][bc.bc_mask_prev] += (1 / 3) * (
                            at_bc * bc.bc_n_vec[j]
                        )
                    elif bc.bc_type == "periodic":

                        rhs_adj[i][bc.bc_mask_prev] += (
                            var()[i][bc.bc_mask_prev]
                            / (2.0 * dx[j])
                            * float(bc.bc_n_dir)
                        )
                    else:
                        # Dirichlet and Symmetry BC: Do nothing
                        pass

        return rhs_adj


def _grad_central_adjust(var: Field, A_ops: tuple[list[Tensor], ...], dim: int) -> None:
    """Function separated from the class to be re-used in Div central scheme.

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
                    Ap[j][dim][bc.bc_mask_prev] = 4 / 3
                    Ac[j][dim][bc.bc_mask_prev] = -4 / 3
                    Am[j][dim][bc.bc_mask_prev] = 0.0
                else:
                    # At upper side
                    Ap[j][dim][bc.bc_mask_prev] = 0.0
                    Ac[j][dim][bc.bc_mask_prev] = -4 / 3
                    Am[j][dim][bc.bc_mask_prev] = 4 / 3
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

        FUTURE: Quick scheme
    """

    def __init__(self):
        self._op_type = __class__.__name__

    @staticmethod
    def build_A_coeffs(
        var_j: Field | float | Tensor, var_i: Field, config: dict[str, dict[str, str]]
    ) -> list[list[Tensor]]:
        r"""Build the coefficients for the discretization of the gradient operator using the second-order central finite difference method.

        ..math::
            \nabla \Phi = \frac{\Phi^{i+1} - \Phi^{i-1}}{2 \Delta x}
        """

        if isinstance(var_j, float):
            adv = torch.ones_like(var_i()) * var_j
        elif isinstance(var_j, Tensor):
            adv = var_j
        else:
            adv = var_j()

        # Shape check
        assert adv.shape == var_i().shape, "FDC Div: adv shape must match var_i shape"

        if config is not None and "limiter" in config["div"]:
            limiter = config["div"]["limiter"]
        else:
            warnings.warn("FDM: no limiter is specified. Use `none` as default.")
            limiter = "none"

        App = [torch.zeros_like(var_i()) for _ in range(var_i.mesh.dim)]
        Ap = [torch.ones_like(var_i()) for _ in range(var_i.mesh.dim)]
        Ac = [torch.zeros_like(var_i()) for _ in range(var_i.mesh.dim)]
        Am = [-1.0 * torch.ones_like(var_i()) for _ in range(var_i.mesh.dim)]
        Amm = [torch.zeros_like(var_i()) for _ in range(var_i.mesh.dim)]

        if limiter == "none":
            Ap, Ac, Am = _adv_central(adv, var_i, [Ap, Ac, Am])
            pass
        elif limiter == "upwind":
            Ap, Ac, Am = _adv_upwind(adv, var_i, [Ap, Ac, Am])
        elif limiter == "quick":
            raise NotImplementedError("FDC Div: quick scheme is not implemented yet.")
        else:
            raise RuntimeError(f"FDC Div: {limiter=} is an unknown limiter type.")

        return [App, Ap, Ac, Am, Amm]

    @staticmethod
    def adjust_rhs(var: Field) -> Tensor:
        ...


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
        _grad_central_adjust(var, (Ap, Ac, Am), i)

    Ap = [adv * ap for ap in Ap]
    Ac = [adv * ac for ac in Ac]
    Am = [adv * am for am in Am]

    A_ops[1] = Ap
    A_ops[2] = Ac
    A_ops[3] = Am

    return A_ops


def _adv_upwind(
    adv: Tensor, var: Field, A_ops: list[list[Tensor]]
) -> list[list[Tensor]]:

    _, Ap, Ac, Am, _ = A_ops

    dx = var.dx

    A_ops[1] = Ap
    A_ops[2] = Ac
    A_ops[3] = Am

    return A_ops


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


@dataclass
class FDC_old:
    """Collection of the operators for explicit finite difference discretization."""

    def ddt(self, dt: float, var: Field) -> Tensor:
        """Time derivative of a given field.

        Note:
            - If `var` does not have old `VARo`, attribute, current `VAR` will be treated as `VARo`.
        """

        ddt = []

        for i in range(var.dim):
            try:
                var_o = var.VARo[i]
            except AttributeError:
                # If not saved, use current value (treat for the first iteration)
                var_o = var()[i]
            ddt.append((var()[i] - var_o) / dt)

        return torch.stack(ddt)

    def div(self, var_j: Field, var_i: Field) -> Tensor:
        """Divergence of two fields.
        Note:
            - To avoid the checkerboard problem, flux limiter is used. It supports `none`, `upwind` and `quick` limiter. (here, `none` is equivalent to the second order central difference.)
        """

        if self.config is not None and "limiter" in self.config["div"]:
            limiter = self.config["div"]["limiter"]
        else:
            warnings.warn("FDM: no limiter is specified. Use `none` as default.")
            limiter = "none"

        if var_j.name == var_i.name:
            # If var_i and var_j are the same field, use the same tensor.
            var_j.set_var_tensor(var_i().clone())

        div = []

        dx = var_j.dx

        for i in range(var_i.dim):

            d_val = torch.zeros_like(var_i()[i])

            for j in range(var_j.dim):

                if limiter == "none":
                    """Central difference scheme."""

                    pad = create_pad(var_i.mesh.dim)
                    slicer = inner_slicer(var_i.mesh.dim)

                    bc_il = var_i.get_bc(f"d-{NUM_TO_DIR[j]}l")
                    bc_ir = var_i.get_bc(f"d-{NUM_TO_DIR[j]}r")

                    # m_val = fill_pad(pad(var_i()[i]), j, 1, slicer)
                    m_val = fill_pad_bc(pad(var_i()[i]), 1, slicer, [bc_il, bc_ir], j)

                    d_val += (
                        var_j()[j]
                        * (
                            (torch.roll(m_val, -1, j) - torch.roll(m_val, 1, j))
                            / (2 * dx[j])
                        )[slicer]
                    )

                elif limiter == "upwind":

                    pad = create_pad(var_i.mesh.dim)
                    slicer = inner_slicer(var_i.mesh.dim)

                    var_i_pad = fill_pad(pad(var_i()[i]), j, 1, slicer)
                    var_j_pad = fill_pad(pad(var_j()[j]), j, 1, slicer)

                    m_val_p = (torch.roll(var_j_pad, -1, j) + var_j_pad) / 2
                    m_val_m = (torch.roll(var_j_pad, 1, j) + var_j_pad) / 2

                    f_val_p = (m_val_p + m_val_p.abs()) * var_i_pad / 2 - (
                        m_val_p - m_val_p.abs()
                    ) * torch.roll(var_i_pad, -1, j) / 2

                    f_val_m = (m_val_m + m_val_m.abs()) * torch.roll(
                        var_i_pad, 1, j
                    ) / 2 - (m_val_p - m_val_p.abs()) * var_i_pad / 2

                    d_val += ((f_val_p - f_val_m) / dx[j])[slicer]

                elif limiter == "quick":
                    pad = create_pad(var_i.mesh.dim, 2)
                    slicer = inner_slicer(var_i.mesh.dim, 2)

                    pass
                else:
                    raise ValueError("FDM: Unknown limiter.")

            div.append(d_val)

        return torch.stack(div)
