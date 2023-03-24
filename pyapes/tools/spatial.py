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

        if var.mesh.coord_sys == "xyz":
            ...

        elif var.mesh.coord_sys == "rz":
            ...
        else:
            raise RuntimeError(f"DiffFlux: unknown coordinate system.")

        return flux


class ScalarOP:
    """Manipulation of a scalar field (scalar operations)

    Note:
        - `jac` and `hess` operations both use the `torch.gradient` function with edge order of 2.
    """

    @staticmethod
    def jac(var: Field) -> Jac:
        assert var().shape[0] == 1, "Scalar: var must be a scalar field."

        jac = FDC.grad(var, edge=True)[0]

        data_jac: dict[str, Tensor] = {}

        if var.mesh.coord_sys == "xyz":
            for i, j in enumerate(jac):
                data_jac[NUM_TO_DIR[i]] = j
        elif var.mesh.coord_sys == "rz":
            for i, j in enumerate(jac):
                data_jac[NUM_TO_DIR_RZ[i]] = j
        else:
            raise RuntimeError(
                f"Spatial: unknown coordinate system: {var.mesh.coord_sys}"
            )
        return Jac(**data_jac)

    @staticmethod
    def hess(var: Field) -> Hess:
        indices = tensor_idx(var.mesh.dim)

        data_hess: dict[str, Tensor] = {}

        hess: list[Tensor] = []

        jac = FDC.grad(var, edge=True)[0]

        hess = [FDC.grad(var.set_var_tensor(j), edge=True)[0] for j in jac]

        for i, hi in enumerate(hess):
            for j, h in enumerate(hi):
                if var.mesh.coord_sys == "xyz":
                    if (i, j) in indices:
                        data_hess[NUM_TO_DIR[i] + NUM_TO_DIR[j]] = h
                elif var.mesh.coord_sys == "rz":
                    if (i, j) in indices:
                        data_hess[NUM_TO_DIR_RZ[i] + NUM_TO_DIR_RZ[j]] = h
                else:
                    raise RuntimeError(
                        f"Spatial: unknown coordinate system: {var.mesh.coord_sys}"
                    )

        return Hess(**data_hess)
