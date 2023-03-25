#!/usr/bin/env python3
"""Cylinder geometry."""
from math import pi

from pyapes.geometry.basis import bound_edge_and_corner
from pyapes.geometry.basis import GeoBounder
from pyapes.geometry.basis import Geometry


class Cylinder(Geometry, metaclass=GeoBounder):
    """Cylinder geometry. Due to the axisymmetric nature of the cylinder, the object is always two dimensional.

    >>> Cylinder([0, 0, 0], [1, 1, 1])       # Option 1
    >>> Cylinder[0:1, 0:1, 0:1]              # Option 2

    Note:
        - Here, leading dimension is the radius (r) and the second dimension is the axis (z).


    Args:
        lower: lower bound of the Box
        upper: upper bound of the Box
    """

    def __init__(
        self,
        lower: list[float] or tuple[float, ...],
        upper: list[float] or tuple[float, ...],
    ):
        assert (
            len(lower) == 2 and len(upper) == 2
        ), "Cylinder: a length of inputs has to be 2 since it is axisymmetric (r-z)!)"

        assert (
            lower[0] >= 0
        ), "Cylinder: lower bound of radius has to be larger (or equal) to 0!"

        # Make sure to be a list and contains float
        self._lower = [float(i) for i in lower]
        self._upper = [float(i) for i in upper]

        # Box element discriminator
        self.ex, self.xp, self.face, self._dim = bound_edge_and_corner(
            self.lower, self.upper, "rz"
        )

        self._config: dict[int, dict[str, list[float] | str]] = {}

        # Create all face configurations
        for idx, (e, x, f) in enumerate(zip(self.ex, self.xp, self.face)):
            self._config.update({idx: {"e_x": e, "x_p": x, "face": f}})

    @property
    def dim(self) -> int:
        """Cylinder dimension."""
        return self._dim

    @property
    def type(self) -> str:
        """Geometry type."""
        return self.__class__.__name__.lower()

    @property
    def size(self) -> float:
        r"""Set size (volume) of the cylinder.

        .. math::
            V = \pi r^2 * z
        """

        self._size = (
            pi * (self.upper[0] - self.lower[0]) ** 2 * (self.upper[1] - self.lower[1])
        )

        return self._size

    @property
    def X(self) -> float:
        return self._lower[0]

    @property
    def Y(self) -> float:
        return self._lower[1]

    @property
    def config(self) -> dict[int, dict[str, list[float] | str]]:
        return self._config

    @property
    def lower(self) -> list[float]:
        return self._lower

    @property
    def upper(self) -> list[float]:
        return self._upper
