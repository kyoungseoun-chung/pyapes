#!/usr/bin/env python3
"""Basis of geometries."""
from enum import Enum
from typing import Any

DIR = ["x", "y", "z"]
DIR_TO_NUM: dict[str, int] = {"x": 0, "y": 1, "z": 2}
FDIR = ["xl", "xr", "yl", "yr", "zl", "zr"]
FDIR_TO_NUM: dict[str, int] = {
    "xl": 0,
    "xr": 1,
    "yl": 2,
    "yr": 3,
    "zl": 4,
    "zr": 5,
}


class Order:
    def __call__(self, f: str):
        return getattr(self, f)


class _DimOrder(Order):
    """Data index: X - 0, Y - 1, Z - 2.

    * Domain is defined as following manner:

        .. code-block:: text

                z (s-n)
                |
                .---- x (w-e)
               /
              y (b-f)
    """

    w = 0
    e = 0
    f = 1
    b = 1
    n = 2
    s = 2


class _NormalDir(Order):
    """Sign of the normal direction."""

    w = -1
    e = 1
    n = 1
    s = -1
    f = 1
    b = -1


class FaceDir(Enum):
    w = 0
    e = 1
    n = 2
    s = 3
    f = 4
    b = 5


NormalDir = _NormalDir()
DimOrder = _DimOrder()


class GeoTypeIdentifier(list):
    """Class that helps to idensitfy the list of types."""

    def __contains__(self, typ: type):
        for val in self:
            if isinstance(val, typ):
                return True
        return False


class Geometry:
    """Base class of all geometries.

    Note:
        - `self.X, self.Y, self.Z` indicate base of geometries. (a cylinder will have center - radius - height instead.)
    """

    @property
    def X(self) -> float:
        raise NotImplementedError

    @property
    def Y(self) -> float:
        raise NotImplementedError

    @property
    def Z(self) -> float:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """dimension of the geometry."""
        raise NotImplementedError

    @property
    def type(self) -> str:
        raise NotImplementedError

    @property
    def size(self) -> float:
        """Geometry size."""
        raise NotImplementedError

    @property
    def lower(self) -> list[float]:
        """Lower bounds of the geometry."""
        raise NotImplementedError

    @property
    def upper(self) -> list[float]:
        """Upper bounds of the geometry."""
        raise NotImplementedError

    @property
    def config(self) -> list[dict]:
        """Configuration of the geometry."""
        raise NotImplementedError

    def __eq__(self, other: Any):

        return (self.lower == other.lower) and (self.size == other.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, size={self.size:.1e})"
