#!/usr/bin/env python3
"""Discretization using finite volume methodology (FVM) """
from typing import Any
from typing import Union

from torch import Tensor

from pyABC.core.variables import Field
from pyABC.core.solver.tools import DIR, FDIR


class Flux:
    """Flux container.

    Note:
        - To distinguis leading index and other index, leading is int and other is str.

    >>> flux_tensor = torch.tensor(...)
    >>> flux = Flux()
    >>> flux.add(i, j, flux_tensor)  # j -> DIR[j] -> "x" or "y" or "z", i -> 0 or 1 or 2.
    >>> flux(0, "x")  # return flux_tensor
    """

    def __init__(self):

        self._center: dict[int, dict[str, Tensor]] = {}
        self._face: dict[int, dict[str, dict[str, Tensor]]] = {}

    def to_center(self, i: int, j: str, T: Tensor):
        """Add centervalued as dictionary."""

        try:
            self._center[i][j] = T
        except KeyError:
            self._center.update({i: {j: T}})

    def __call__(self, i: int, j: str) -> Tensor:
        """Return flux values with parentheses."""

        if j in DIR:
            return self._center[i][j]
        else:
            assert j in FDIR, f"Flux: face index should be one of {FDIR}"
            return self._face[i][j[0]][j[1]]

    @property
    def c_idx(self) -> tuple[list[int], list[str]]:
        """Return center index."""

        idx_i = list(self._center.keys())
        idx_j = list(self._center[0].keys())

        return (idx_i, idx_j)

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

    def flux_sum(self) -> None:

        self._center = {}

        for i in self._face:
            c_val = {}
            for j in self._face[i]:
                c_val.update(
                    {j: (self._face[i][j]["l"] + self._face[i][j]["r"]) / 2}
                )
            self._center[i] = c_val

    def __mul__(self, target: Union[float, int, Field]) -> Any:
        """Multiply coeffcient to the flux"""

        if isinstance(target, float) or isinstance(target, int):
            for i in self._face:
                for j in self._face[i]:
                    self._face[i][j]["l"] *= target
                    self._face[i][j]["r"] *= target
                    try:
                        self._center[i][j] *= target
                    except KeyError:
                        self.flux_sum()
                        self._center[i][j] *= target

        elif isinstance(target, Field):
            # Multiply tensor in j direction for self._center
            # Will be used for u_j \nabla_j u_i
            for i in self._face:
                for j in self._face[i]:
                    self._center[i][j] *= target()
        else:
            raise TypeError("Flux: wrong input type is given!")

        return self
