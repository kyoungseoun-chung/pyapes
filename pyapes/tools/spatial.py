#!/usr/bin/env python3
"""Tools that manipulate spatial data."""

from dataclasses import dataclass
from pyapes.core.variables import Field
from pyapes.core.geometry.basis import NUM_TO_DIR, NUM_TO_DIR_RZ
from pymytools.indices import tensor_idx
import torch
from torch import Tensor


class Derivatives:
    def __init__(self):

        self.max = 0

        total_var = len(vars(self).items()) - 1

        for idx, (_, v) in enumerate(vars(self).items()):

            if idx == total_var:
                # Exclude self.max for counting
                break

            if v is None:
                pass
            else:
                self.max += 1

        self.keys = [
            k
            for idx, (k, v) in enumerate(vars(self).items())
            if idx < total_var and v is not None
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

    x: Tensor | None = None
    y: Tensor | None = None
    z: Tensor | None = None
    r: Tensor | None = None

    def __post_init__(self):
        super().__init__()


@dataclass
class Hess(Derivatives):

    xx: Tensor | None = None
    xy: Tensor | None = None
    xz: Tensor | None = None
    yy: Tensor | None = None
    yz: Tensor | None = None
    zz: Tensor | None = None
    rr: Tensor | None = None
    rz: Tensor | None = None
    zz: Tensor | None = None

    def __post_init__(self):
        super().__init__()


class ScalarOP:
    """Manipulation of a scalar field (scalar operations)

    Note:
        - `jac` and `hess` operations both use the `torch.gradient` function with edge order of 2.
    """

    @staticmethod
    def jac(var: Field) -> Jac:

        assert var().shape[0] == 1, "Scalar: var must be a scalar field."

        jac = torch.gradient(var()[0], spacing=var.mesh.dx.tolist(), edge_order=2)

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

        jac = __class__.jac(var)

        indices = tensor_idx(var.mesh.dim)

        hess: list[list[Tensor]] = []

        data_hess: dict[str, Tensor] = {}
        hess = [
            torch.gradient(j, spacing=var.mesh.dx.tolist(), edge_order=2) for j in jac
        ]

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
