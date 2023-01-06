#!/usr/bin/env python3
"""Basis of geometries."""
from typing import Any
from typing import Union

DIR = ["x", "y", "z"]
DIR_TO_NUM: dict[str, int] = {"x": 0, "y": 1, "z": 2}
NUM_TO_DIR: dict[int, str] = {0: "x", 1: "y", 2: "z"}
FDIR = ["xl", "xr", "yl", "yr", "zl", "zr"]  # could be used for bc identifier?
FDIR_TO_NUM: dict[str, int] = {
    "xl": 0,
    "xr": 1,
    "yl": 2,
    "yr": 3,
    "zl": 4,
    "zr": 5,
}


class GeoTypeIdentifier(list):
    """Class that helps to identify the list of types."""

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
    def config(self) -> dict[int, dict[str, Union[list[list[float]], str]]]:
        """Configuration of the geometry."""
        raise NotImplementedError

    def __eq__(self, other: Any):

        return (self.lower == other.lower) and (self.size == other.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, size={self.size:.1e})"
