#!/usr/bin/env python3
"""Tools that manipulate spatial data.
Mostly for the discretization without boundary treatment.
"""
from dataclasses import dataclass

import torch
from pymytools.indices import tensor_idx
from torch import Tensor

from pyapes.core.geometry.basis import NUM_TO_DIR
from pyapes.core.geometry.basis import NUM_TO_DIR_RZ
from pyapes.core.solver.fdc import FDC
from pyapes.core.variables import Field


class Derivatives:
    """Base class for Jacobian and Hessian. Intention is to use generic indices (x, y, z for example) to access each derivative group.

    Example:
        >>> jac = Jac(x=...)
        >>> jac.x
        Tensor(...)
        >>> hess = Hess(xx=...)
        >>> hess.xx
        Tensor(...)
    """

    def __init__(self):
        self.max = 0

        total_var = len(vars(self).items()) - 1

        for idx, (_, v) in enumerate(vars(self).items()):
            if idx == total_var:
                # Exclude self.max for counting
                break

            if v.shape[0] == 0:
                pass
            else:
                self.max += 1

        self.keys = [
            k
            for idx, (k, v) in enumerate(vars(self).items())
            if idx < total_var and v.shape[0] != 0
        ]

    def __getitem__(self, key: str) -> Tensor:
        """Return the derivative group by a key. If the key is given for the Hessian, the key is always sorted in alphabetical order.

        Example:
            >>> hess["xz"]
            hess.xz
            >>> hess["zx"]
            hess.xz
            >>> hess["yx"]
            hess.xy
        """

        item = getattr(self, "".join(sorted(key.lower())))
        if item.shape[0] == 0:
            raise KeyError(f"Derivative: key {key} not found.")
        else:
            return item

    def __len__(self) -> int:
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            res = getattr(self, self.keys[self.n])
            self.n += 1
            return res
        else:
            raise StopIteration


@dataclass
class Jac(Derivatives):
    x: Tensor = torch.tensor([])
    y: Tensor = torch.tensor([])
    z: Tensor = torch.tensor([])
    r: Tensor = torch.tensor([])

    def __post_init__(self):
        super().__init__()


@dataclass
class Hess(Derivatives):
    xx: Tensor = torch.tensor([])
    xy: Tensor = torch.tensor([])
    xz: Tensor = torch.tensor([])
    yy: Tensor = torch.tensor([])
    yz: Tensor = torch.tensor([])
    zz: Tensor = torch.tensor([])
    rr: Tensor = torch.tensor([])
    rz: Tensor = torch.tensor([])
    zz: Tensor = torch.tensor([])

    def __post_init__(self):
        super().__init__()


class DiffFlux:
    """Object to be used in the tensor diffussion term."""

    def __new__(cls, diff: Hess, var: Field):
        # ignore the args
        return cls.__call__(diff, var)

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

        jac = ScalarOP.jac(var)
        flux = Field("DiffFlux", len(jac), var.mesh, None)

        n2d = _n2d_coord(var.mesh.coord_sys)

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


class ScalarOP:
    """Manipulation of a scalar field (scalar operations)

    Note:
        - `jac` and `hess` operations both use the `torch.gradient` function with edge order of 2.
    """

    @staticmethod
    def jac(var: Field) -> Jac:
        assert var().shape[0] == 1, "Scalar: var must be a scalar field."

        data_jac: dict[str, Tensor] = {}

        n2d = _n2d_coord(var.mesh.coord_sys)

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

        n2d = _n2d_coord(var.mesh.coord_sys)

        jac = FDC.grad(var, edge=True)[0]

        jac_f = var.copy()

        hess = [FDC.grad(jac_f.set_var_tensor(j), edge=True)[0] for j in jac]

        for i, hi in enumerate(hess):
            for j, h in enumerate(hi):
                if (i, j) in indices:
                    data_hess[n2d[i] + n2d[j]] = h

        FDC.grad.reset()
        return Hess(**data_hess)


def _n2d_coord(coord: str) -> dict[int, str]:
    if coord == "xyz":
        n2d = NUM_TO_DIR
    elif coord == "rz":
        n2d = NUM_TO_DIR_RZ
    else:
        raise RuntimeError(f"DiffFlux: unknown coordinate system.")

    return n2d
