#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM) """
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import torch
from torch import Tensor

from pyapes.core.geometry.basis import DIR
from pyapes.core.geometry.basis import FDIR
from pyapes.core.mesh import Mesh


F_c_type = dict[int, dict[str, Tensor]]
F_f_type = dict[int, dict[str, dict[str, Tensor]]]


@dataclass
class Flux:
    """Flux container.

    Note:
        - To distinguish leading index and other index, leading is int and other is str.

    >>> flux_tensor = torch.tensor(...)
    >>> flux = Flux()
    >>> flux.to_face(i, j, d, flux_tensor)  # j -> DIR[j] -> "x" or "y" or "z", i -> 0 or 1 or 2. d -> "l" or "r"
    >>> flux(0, "x")  # return Flux._sum
    >>> flux(0, "xl") # return Flux._face
    """

    mesh: Mesh
    _center: F_c_type = field(default_factory=dict)
    _face: F_f_type = field(default_factory=dict)

    def __call__(self, i: int, j: str) -> Tensor:
        """Return flux values with parentheses."""

        if j in DIR:
            return self.center(i, j)
        else:
            return self.face(i, j)

    def tensor(self, f_all: bool = False) -> Tensor:
        """Return `self._center` as Tensor.

        Args:
            f_all (bool, optional): If True, tensor is summed over mesh coordinate, Else, keep its dimensions.
        """

        var_tensor = []
        for v_d in self._center:
            if f_all:
                var_tensor.append(self._center[v_d]["all"])
            else:
                coord_tensor = []
                for c_d in self._center[v_d]:
                    coord_tensor.append(self._center[v_d][c_d])
                var_tensor.append(torch.stack(coord_tensor, dim=0))

        return torch.stack(var_tensor)

    def center(self, i: int, s_idx: str) -> Tensor:
        """Return flux sum."""
        assert s_idx in DIR, f"Flux: sum index should be one of {DIR}!"

        return self._center[i][s_idx]

    def face(self, i: int, f_idx: str) -> Tensor:
        """Return face value with index."""

        assert f_idx in FDIR, f"Flux: face index should be one of {FDIR}!"

        return self._face[i][f_idx[0]][f_idx[1]]

    def to_face(self, i: int, j: str, f: str, T: Tensor) -> None:
        """Assign face values to `self._face`.

        Args:
            i (int): leading index
            j (str): dummy index (to be summed)
            f (str): face index l (also for back and bottom), r (also for front and top)
            T (Tensor): face values to be stored.
        """

        if i in self._face:
            if j in self._face[i]:
                self._face[i][j][f] = T
            else:
                self._face[i].update({j: {f: T}})
        else:
            self._face.update({i: {j: {f: T}}})

    def to_center(self, i: int, j: str, T: Tensor):
        """Assign face values to `self._center`.

        Args:
            i (int): leading index
            j (str): dummy index (to be summed)
            T (Tensor): center values to be stored.
        """

        if i in self._center:
            if j in self._center[i]:
                self._center[i][j] = T
            else:
                self._center[i].update({j: T})
        else:
            self._center.update({i: {j: T}})

    def sum_all(self) -> None:
        """Sum fluxes in all directions."""

        for i in self._face:
            val_tensor = torch.zeros_like(self.mesh.V)
            for j in self._face[i]:
                Al_V = self.mesh.A[j + "l"] / self.mesh.V
                Ar_V = self.mesh.A[j + "r"] / self.mesh.V

                # sumAll face_val * face_are / cell_volume
                val_tensor += (
                    self._face[i][j]["r"] * Ar_V - self._face[i][j]["l"] * Al_V
                )

            self._center[i] = {"all": val_tensor}

    def sum(self) -> None:
        """Sum fluxes in each direction at the faces and assign to the center of a cell volume."""

        for i in self._face:
            c_val = {}

            for j in self._face[i]:
                Al_V = self.mesh.A[j + "l"] / self.mesh.V
                Ar_V = self.mesh.A[j + "r"] / self.mesh.V

                # sumAll face_val * face_are / cell_volume
                c_val.update(
                    {j: (self._face[i][j]["r"] * Ar_V - self._face[i][j]["l"] * Al_V)}
                )
            self._center[i] = c_val

    def limiter(self, l_type: str) -> None:
        raise NotImplementedError

    def __mul__(self, target: float | int) -> Flux:
        """Multiply coefficient to the flux"""

        for i in self._face:
            for j in self._face[i]:
                self._face[i][j]["l"] *= target
                self._face[i][j]["r"] *= target

        for i in self._center:
            for j in self._center[i]:
                self._center[i][j] *= target

        return self
