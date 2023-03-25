#!/usr/bin/env python3
"""Box geometry."""
#!/usr/bin/env python3
from pyapes.geometry.basis import bound_edge_and_corner
from pyapes.geometry.basis import GeoBounder
from pyapes.geometry.basis import Geometry


BOX_DIM = [1, 2, 3]


class Box(Geometry, metaclass=GeoBounder):
    """Box geometry.

    >>> Box([0, 0, 0], [1, 1, 1])       # Option 1
    >>> Box[0:1, 0:1, 0:1]              # Option 2

    Note:
        - Box `lower` and `upper` is the list of either ints or floats. However,
          once the class is instantiated, given bounds will be converted to
          the list of floats.

    Args:
        lower: lower bound of the Box
        upper: upper bound of the Box
    """

    def __init__(
        self,
        lower: list[float] or tuple[float, ...],
        upper: list[float] or tuple[float, ...],
    ):
        assert len(lower) == len(upper), "Box: length of inputs has to be matched!"

        # Make sure to be a list and contains float
        self._lower = [float(i) for i in lower]
        self._upper = [float(i) for i in upper]

        # Box element discriminator
        self.ex, self.xp, self.face, self._dim = bound_edge_and_corner(
            self.lower, self.upper
        )

        self._config: dict[int, dict[str, list[float] | str]] = {}

        # Create all face configurations
        for idx, (e, x, f) in enumerate(zip(self.ex, self.xp, self.face)):
            self._config.update({idx: {"e_x": e, "x_p": x, "face": f}})

    @property
    def dim(self) -> int:
        """Box dimension."""
        return self._dim

    @property
    def type(self) -> str:
        """Geometry type."""
        return self.__class__.__name__.lower()

    @property
    def size(self) -> float:
        """Set size of the box."""

        self._size = 1.0
        for l, u in zip(self.lower, self.upper):
            self._size *= float(u - l)

        return self._size

    @property
    def X(self) -> float:
        return self._lower[0]

    @property
    def Y(self) -> float:
        return self._lower[1]

    @property
    def Z(self) -> float:
        return self._lower[2]

    @property
    def config(self) -> dict[int, dict[str, list[float] | str]]:
        return self._config

    @property
    def lower(self) -> list[float]:
        return self._lower

    @property
    def upper(self) -> list[float]:
        return self._upper
