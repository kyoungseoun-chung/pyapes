#!/usr/bin/env python3
from typing import cast
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyapes.core.backend import DType
from pyapes.core.backend import DTYPE_DOUBLE
from pyapes.core.backend import DTYPE_SINGLE
from pyapes.core.backend import TORCH_DEVICE
from pyapes.core.backend import TorchDevice
from pyapes.core.geometry import GeoTypeIdentifier
from pyapes.core.geometry.basis import Geometry


class Mesh:
    """Equidistance rectangular (in mind) mesh.

    Args:
        domain (Geometry): Domain geometry.
        obstacle (Optional[list[Geometry]]): Obstacle geometry.
        spacing (Union[list[int], list[float]], optional): Mesh spacing. Defaults to [].
        device (str, optional): Device for `torch.Tensor`. Defaults to "cpu".
        dtype (Union[str, int], optional): Data type for `torch.Tensor`. Defaults to "double".
    """

    def __init__(
        self,
        domain: Geometry,
        obstacle: Optional[list[Geometry]],
        spacing: Union[list[int], list[float]] = [],
        device: str = "cpu",
        dtype: Union[str, int] = "double",
    ):

        assert device in TORCH_DEVICE
        self.device = TorchDevice(device).device

        assert dtype in DTYPE_DOUBLE or dtype in DTYPE_SINGLE
        self.dtype = DType(dtype)

        self.domain = domain
        self.obstacle = obstacle

        self._lower = torch.tensor(
            self.domain.lower,
            dtype=self.dtype.float,
            device=self.device.type,
            requires_grad=False,
        )
        self._upper = torch.tensor(
            self.domain.upper,
            dtype=self.dtype.float,
            device=self.device.type,
            requires_grad=False,
        )
        self._lx = self._upper - self._lower

        if int in GeoTypeIdentifier(spacing):
            # Node information
            self._nx: list[int] = [int(s) for s in spacing]
            self._dx: list[float] = [
                float(l / (n - 1.0)) for l, n in zip(self._lx, self._nx)
            ]

        elif float in GeoTypeIdentifier(spacing):
            # Node information
            self._dx: list[float] = [float(s) for s in spacing]
            self._nx: list[int] = [
                int(l / d + 1.0) for l, d in zip(self._lx, self._dx)
            ]

        else:
            raise TypeError("Mesh: spacing only accept int or float")

        self.x = []

        for i in range(self.dim):
            self.x.append(
                torch.linspace(
                    self.lower[i].item(),
                    self.upper[i].item(),
                    self.nx[i],
                    dtype=self.dtype.float,
                    device=self.device,
                )
            )

        # Mesh grid
        self.grid = torch.meshgrid(self.x, indexing="ij")
        """Mesh grid created by `torch.meshgrid`"""

        # Obtain face area and volume
        self._A, self._V = self.get_A_and_V()

        # Mesh mask
        self.d_mask, self.o_mask = boundary_mask(self)

        self.t_mask = torch.zeros_like(self.d_mask["xl"])
        """Mask combined all."""

        # Get all mask
        for dm in self.d_mask:
            self.t_mask = torch.logical_or(self.t_mask, self.d_mask[dm])

        if len(self.o_mask) > 0:
            for o_idx in self.o_mask:
                for om in self.o_mask[o_idx]:
                    self.t_mask = torch.logical_or(
                        self.t_mask, self.o_mask[o_idx][om]
                    )

    @property
    def _depth(self) -> float:
        """Depth of mesh. If the mesh is 1D or 2D, takes `self.dx[0]` as a reference size."""

        if self.dim == 1:
            depth = self.dx[0].item() * self.dx[0].item()
        elif self.dim == 2:
            depth = self.dx[0].item()
        else:
            depth = 1.0
        return depth

    @property
    def A(self) -> dict[str, Tensor]:
        """Area of the cell."""
        return self._A

    @property
    def V(self) -> Tensor:
        """Volume of the cell."""
        return self._V

    def get_A_and_V(self) -> tuple[dict[str, Tensor], Tensor]:
        """Calculate cell area and volume."""

        return self.get_A(), self.get_V()

    def get_A(self) -> dict[str, Tensor]:
        """Area of mesh.
        Leading dimension is `2 * self.mesh.dim`. Therefore comes with the following order: 0 -> x-l, 1 -> x-r, 2 -> y-l, 3 -> y-r, 4 -> z-l, 5 -> z-r.
        """

        faces = 2 * self.dim
        area = torch.zeros(
            (faces, *self.nx), dtype=self.dtype.float, device=self.device
        )

        if self.dim == 1:
            area += self._depth
            return {"xl": area[0], "xr": area[1]}
        elif self.dim == 2:
            area[:2, :] = self.dx[1] * self._depth
            area[2:, :] = self.dx[0] * self._depth
            return {"xl": area[0], "xr": area[1], "yl": area[2], "yr": area[3]}
        else:
            area[:2, :] = self.dx[1] * self.dx[2]
            area[2:4, :] = self.dx[0] * self.dx[2]
            area[4:, :] = self.dx[0] * self.dx[1]
            return {
                "xl": area[0],
                "xr": area[1],
                "yl": area[2],
                "yr": area[3],
                "zl": area[4],
                "zr": area[5],
            }

    def get_V(self) -> Tensor:
        """Volume of mesh."""

        vol = torch.zeros(*self.nx, dtype=self.dtype.float, device=self.device)

        if self.dim == 1:
            return vol + self.dx[0] * self._depth
        elif self.dim == 2:
            return vol + self.dx[0] * self.dx[1] * self._depth
        else:
            return vol + self.dx[0] * self.dx[1] * self.dx[2]

    @property
    def dim(self) -> int:
        """Dimension of the mesh."""
        return self.domain.dim

    @property
    def X(self) -> Tensor:
        """Return X coordinate of the mesh."""
        return self.grid[0]

    @property
    def Y(self) -> Tensor:
        """Return Y coordinate of the mesh.
        Note:
            If `self.dim > 1` return empty tensor.
        """
        return (
            self.grid[1]
            if self.dim > 1
            else torch.tensor([], dtype=self.dtype.float, device=self.device)
        )

    @property
    def Z(self) -> Tensor:
        """Return Z coordinate of the mesh.
        Note:
            If `self.dim > 2` return empty tensor.
        """
        return (
            self.grid[2]
            if self.dim > 2
            else torch.tensor([], dtype=self.dtype.float, device=self.device)
        )

    @property
    def N(self) -> int:
        """Return total number of grid points."""

        return int(
            torch.prod(
                torch.tensor(
                    self._nx, dtype=self.dtype.int, device=self.device
                )
            )
        )

    @property
    def size(self) -> float:
        """Total volume."""

        return torch.prod(self.upper - self.lower).item()

    @property
    def lx(self) -> Tensor:
        """Domain size."""
        return self._lx

    @property
    def dx(self) -> Tensor:
        """Mesh spacing."""
        return torch.tensor(
            self._dx, dtype=self.dtype.float, device=self.device
        )

    @property
    def nx(self) -> torch.Size:
        """Return number of grid points."""
        return torch.Size(self._nx)

    @property
    def lower(self) -> Tensor:
        """Origin of the mesh coordinates."""
        return self._lower

    @property
    def upper(self) -> Tensor:
        """Origin of the mesh coordinates."""
        return self._upper

    @property
    def center(self) -> Tensor:
        """Center of the mesh."""
        return self.lx * 0.5

    @property
    def is_cuda(self) -> bool:
        """True if cuda is available."""
        return self.device == torch.device("cuda")


def boundary_mask(mesh: Mesh) -> tuple[dict, dict]:
    """Create a mask from the objects (self.mesh.objs).

    Warning:
        - Currently only works for the Patch. For the :py:class:`InnerObject`,
          need separate treatment later

    Returns
        Created mask dictionary. Dictionary key is obj.id.

    """
    x = mesh.x
    dx = mesh.dx
    nx = mesh.nx
    dim = mesh.dim

    # {0: {"e_x": ..., "x_p": ..., "face": ...}, ...}
    domain = mesh.domain
    obstacle = mesh.obstacle

    dtype = mesh.dtype
    device = mesh.device

    domain_mask: dict[str, Tensor] = {}
    object_mask = {}

    # Loop over all faces
    for obj in domain.config:

        mask = torch.zeros(*nx, dtype=dtype.bool, device=device)
        mask = get_box_mask(x, dx, domain.config[obj], mask, dim)

        # Save as sub dictionary
        mask_face_id = cast(str, domain.config[obj]["face"])
        domain_mask.update({mask_face_id: mask})

    # For the inner objects
    if obstacle is not None:

        for i, obj in enumerate(obstacle):
            obj_mask = {}
            if obj.type == "box":
                for o in obj.config:
                    mask = torch.zeros(*nx, dtype=dtype.bool, device=device)
                    mask = get_box_mask(x, dx, obj.config[o], mask, dim)

                    # Save as sub dictionary
                    # Should id be face dir?
                    # Save as sub dictionary
                    mask_face_id = cast(str, obj.config[o]["face"])

                    obj_mask.update({mask_face_id: mask})
                object_mask.update({i: obj_mask})
            else:

                raise NotImplementedError(
                    "Mask: non box type inner obstacle is not supported yet!"
                )

    return domain_mask, object_mask


def get_box_mask(
    x: list[Tensor],
    dx: Tensor,
    obj: dict[str, Union[list[list[float]], str]],
    mask: Tensor,
    dim: int,
) -> Tensor:
    """Get masks for boundaries."""

    _nx = torch.zeros(dim, dtype=torch.long)
    _ix = torch.zeros_like(_nx)

    x_p = torch.tensor(obj["x_p"], dtype=x[0].dtype, device=x[0].device)
    e_x = torch.tensor(obj["e_x"], dtype=x[0].dtype, device=x[0].device)

    for i in range(dim):
        x_p[i] = x[i][torch.argmin(abs(x[i] - x_p[i]))]
        _nx[i] = torch.ceil(e_x[i] / dx[i]).type(torch.long) + 1
        _ix[i] = torch.argmin(abs(x[i] - x_p[i]))

    if dim == 1:
        mask[
            _ix[0] : _ix[0] + _nx[0],
        ] = True
    elif dim == 2:
        mask[
            _ix[0] : _ix[0] + _nx[0],
            _ix[1] : _ix[1] + _nx[1],
        ] = True
    else:
        mask[
            _ix[0] : _ix[0] + _nx[0],
            _ix[1] : _ix[1] + _nx[1],
            _ix[2] : _ix[2] + _nx[2],
        ] = True

    return mask
