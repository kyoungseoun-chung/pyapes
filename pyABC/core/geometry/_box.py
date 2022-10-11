#!/usr/bin/env python3
"""Box geometry."""
#!/usr/bin/env python3
from pyABC.core.geometry import Geometry


BOX_DIM = [1, 2, 3]


class BoxType(type):
    """
    Convenience function for creating N-dimensional boxes / cuboids.

    Examples to create a box from (0, 0) to (10, 20):

    * box[0:10, 0:20]
    * box((0, 0), (10, 20))
    """

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = [item]
        lower = []
        upper = []
        for dim in item:
            assert isinstance(dim, slice)
            assert type(dim.start) is float or type(dim.start) is int
            assert type(dim.stop) is float or type(dim.stop) is int
            assert dim.step is None, "Box: step must be None"

            # Lower and upper are forced to have float datatype
            lower.append(float(dim.start))
            upper.append(float(dim.stop))

        return Box(lower, upper)


class Box(Geometry, metaclass=BoxType):
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

        assert len(lower) == len(
            upper
        ), "Box: length of inputs has to be matched!"

        # Make sure to be a list and contains float
        self._lower = [float(i) for i in lower]
        self._upper = [float(i) for i in upper]

        # Box element discriminator
        self.ex, self.xp, self.id, self.face, self._dim = _box_edge_and_corner(
            self.lower, self.upper
        )

        self._config = []

        # Create object configuration
        for n, e, x, f in zip(self.id, self.ex, self.xp, self.face):

            self._config.append(
                {
                    "name": n,
                    "type": "patch",
                    "geometry": {
                        "e_x": e,
                        "x_p": x,
                        "face": f,
                    },
                }
            )

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

        if self.dim == 1:
            self._size = float(self.upper[0] - self.lower[0])
        elif self.dim == 2:
            self._size = float(
                (self.upper[0] - self.lower[0])
                * (self.upper[1] - self.lower[1])
            )
        else:
            self._size = float(
                (self.upper[0] - self.lower[0])
                * (self.upper[1] - self.lower[1])
                * (self.upper[2] - self.lower[2])
            )

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
    def config(self) -> list[dict]:
        return self._config

    @property
    def lower(self) -> list[float]:
        return self._lower

    @property
    def upper(self) -> list[float]:
        return self._upper


def _box_edge_and_corner(
    lower: list[float], upper: list[float]
) -> tuple[list, list, list, list, int]:
    """Crate edge and  corner information based on input dimension.

    Note:
        - Order is always from - to + (in terms of the normal vector).
        - lower and upper contains location in the order of x - y - z.
        - Currently hard-coded. Any better ideas?

    Args:
        lower: lower bounds of the box
        upper: upper bounds of the box
        dim: box dimension
    """

    dim = len(lower)

    assert dim > 0 and dim < 4, "Box: dimensions must be 1, 2 and 3!"

    if dim == 1:
        # 1D edge and corner
        xp = [[lower[0]], [upper[0]]]
        ex = [[lower[0] - xp[0][0]], [upper[0] - xp[1][0]]]
        id = ["-", "+"]
        face = ["w", "e"]
    elif dim == 2:
        # 2D edge and corner
        xp = [
            [lower[0], lower[1]],
            [lower[0], upper[1]],
            [lower[0], lower[1]],
            [upper[0], lower[1]],
        ]
        ex = [
            [upper[0] - xp[0][0], lower[1] - xp[0][1]],
            [upper[0] - xp[1][0], upper[1] - xp[1][1]],
            [lower[0] - xp[2][0], upper[1] - xp[2][1]],
            [upper[0] - xp[3][0], upper[1] - xp[3][1]],
        ]
        id = ["y-", "y+", "x-", "x+"]
        face = ["b", "f", "w", "e"]
    else:
        # 3D edge and corner
        # Set of xp
        xp = [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [lower[0], lower[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], lower[2]],
            [lower[0], lower[1], upper[2]],
        ]
        ex = [
            [lower[0] - xp[0][0], upper[1] - xp[0][1], upper[2] - xp[0][2]],
            [upper[0] - xp[1][0], upper[1] - xp[1][1], upper[2] - xp[1][2]],
            [upper[0] - xp[2][0], lower[1] - xp[2][1], upper[2] - xp[2][2]],
            [upper[0] - xp[3][0], upper[1] - xp[3][1], upper[2] - xp[3][2]],
            [upper[0] - xp[4][0], upper[1] - xp[4][1], lower[2] - xp[4][2]],
            [upper[0] - xp[5][0], upper[1] - xp[5][1], upper[2] - xp[5][2]],
        ]
        id = ["yz-", "yz+", "xz-", "xz+", "xy-", "xy+"]
        face = ["w", "e", "s", "n", "b", "f"]

    return ex, xp, id, face, dim
