#!/usr/bin/env python3
from functools import cached_property
from typing import cast
from typing import Optional

import torch
from torch import Tensor

from pyapes.backend import DType
from pyapes.backend import DTYPE_DOUBLE
from pyapes.backend import DTYPE_SINGLE
from pyapes.backend import TORCH_DEVICE
from pyapes.backend import TorchDevice
from pyapes.geometry import GeoTypeIdentifier
from pyapes.geometry.basis import DIR_TO_NUM
from pyapes.geometry.basis import Geometry


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
        spacing: list[int] | list[float] = [],
        device: str = "cpu",
        dtype: str | int = "double",
    ):
        assert device in TORCH_DEVICE, "Mesh: device only accept cpu or cuda"
        self.device = TorchDevice(device).device

        assert (
            dtype in DTYPE_DOUBLE or dtype in DTYPE_SINGLE
        ), "Mesh: dtype only accept double or single"
        self.dtype = DType(dtype)

        self.domain = domain

        if self.coord_sys == "rz":
            assert self.dim == 2, "Mesh: rz coordinate system only accept 2D domain"

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
            self._nx: list[int] = [int(l / d + 1.0) for l, d in zip(self._lx, self._dx)]

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

        # Mesh mask
        self.d_mask, self.o_mask = boundary_mask(self)

        self.t_mask = (
            torch.zeros_like(self.d_mask["xl"])
            if self.coord_sys == "xyz"
            else torch.zeros_like(self.d_mask["rl"])
        )
        """Mask combined all."""

        # Get all mask
        for dm in self.d_mask:
            self.t_mask = torch.logical_or(self.t_mask, self.d_mask[dm])

        if len(self.o_mask) > 0:
            for o_idx in self.o_mask:
                for om in self.o_mask[o_idx]:
                    self.t_mask = torch.logical_or(self.t_mask, self.o_mask[o_idx][om])

    def __repr__(self) -> str:
        desc = f"{self.domain} with dx={self.dx.tolist()}"
        return desc

    @property
    def coord_sys(self) -> str:
        """Coordinate system of the domain. e.g. `xyz` (Cartesian), `rz` (Axisymmetric)."""

        if self.domain.type == "box":
            return "xyz"
        elif self.domain.type == "cylinder":
            return "rz"
        else:
            raise TypeError(f"Mesh: domain type ({self.domain.type=}) not identifiable")

    def d_mask_dim(self, d_face: str) -> int:
        """Return the mask face dimension."""

        return DIR_TO_NUM[d_face[0]]

    def d_mask_dir(self, d_face: str) -> int:
        """Return the mask face dir."""

        return 1 if d_face[1] == "r" else -1

    def d_mask_shift(self, d_face: str, shift: int) -> Tensor:
        """Shift the domain mask towrad inner side.
        If `d_face` is "xl", the mask will be shifted to the left.

        Example:
            >>> d_mask = {
                "xl": torch.tensor([True, False, False, False, False]),
                "xr": torch.tensor([False, False, False, False, True])
                }
            >>> d_mask_shift("xl", 1)
            torch.tensor([False, True, False, False, False])
            >>> d_mask_shift("xr", 1)
            torch.tensor([False, False, False, True, False])

        """
        face_dim = self.d_mask_dim(d_face)
        face_dir = self.d_mask_dir(d_face)

        if self.device == torch.device("mps"):
            return (
                torch.roll(
                    self.d_mask[d_face].to(device=torch.device("cpu")),
                    -shift * face_dir,
                    face_dim,
                )
            ).to(device=torch.device("mps"))
        else:
            return torch.roll(self.d_mask[d_face], -shift * face_dir, face_dim)

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
    def dim(self) -> int:
        """Dimension of the mesh. e.g. (x, y, z) -> 3"""
        return self.domain.dim

    @property
    def R(self) -> Tensor:
        """Return R coordinate of the mesh."""
        if self.coord_sys == "xyz":
            raise KeyError("Mesh: R coordinate only available in axisymmetric case.")
        elif self.coord_sys == "rz":
            return self.grid[0]
        else:
            raise NotImplementedError(f"Mesh: {self.coord_sys=} not implemented.")

    @property
    def X(self) -> Tensor:
        """Return X coordinate of the mesh.
        In the axisymmetric case, this will represent radial direction.
        """
        return self.grid[0]

    @property
    def Y(self) -> Tensor:
        """Return Y coordinate of the mesh.
        Note:
            - If `self.coord_sys == "rz"` return empty tensor.
            - If `self.dim > 1` return empty tensor.
        """
        if self.coord_sys == "xyz":
            return (
                self.grid[1]
                if self.dim > 1
                else torch.tensor([], dtype=self.dtype.float, device=self.device)
            )
        else:
            return torch.tensor([], dtype=self.dtype.float, device=self.device)

    @property
    def Z(self) -> Tensor:
        """Return Z coordinate of the mesh.
        Note:
            - If `self.coord_sys == "rz"` return Y coordinate.
            - If `self.dim > 2` return empty tensor.
        """
        if self.coord_sys == "xyz":
            return (
                self.grid[2]
                if self.dim > 2
                else torch.tensor([], dtype=self.dtype.float, device=self.device)
            )
        else:
            return self.grid[1]

    @property
    def N(self) -> int:
        """Return total number of grid points."""

        return int(
            torch.prod(torch.tensor(self._nx, dtype=self.dtype.int, device=self.device))
        )

    @property
    def size(self) -> float:
        """Total volume of the domain."""

        return self.domain.size

    @property
    def lx(self) -> Tensor:
        """Domain size."""
        return self._lx

    @property
    def dx(self) -> Tensor:
        """Mesh spacing."""
        return torch.tensor(self._dx, dtype=self.dtype.float, device=self.device)

    @cached_property
    def dg(self) -> list[Tensor]:
        """Boundary treated grid. `gd` stands for `del grid`."""

        del_grid: list[Tensor] = []

        for idx, g in enumerate(self.grid):
            g_del = torch.zeros_like(g)

            if self.device == torch.device("mps"):
                # MPS does not support torch.roll. Therefore, convert to CPU and back.
                g_rp = (
                    torch.roll(g.clone().to(device=torch.device("cpu")), -1, idx)
                    - g.to(device=torch.device("cpu"))
                ).to(device=torch.device("mps"))
                g_rm = (
                    g.to(device=torch.device("cpu"))
                    - torch.roll(g.clone().to(device=torch.device("cpu")), 1, idx)
                ).to(device=torch.device("mps"))

            else:
                g_rp = torch.roll(g.clone(), -1, idx) - g
                g_rm = g - torch.roll(g.clone(), 1, idx)

            g_rp[g_rp.lt(0.0)] = 0.0
            g_rm[g_rm.lt(0.0)] = 0.0

            g_del += g_rp + g_rm

            del_grid.append(g_del / 2)

        return del_grid

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
            if obj.type == "box" or obj.type == "cylinder":
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
    obj: dict[str, list[list[float]] | str],
    mask: Tensor,
    dim: int,
) -> Tensor:
    """Get masks for boundaries."""

    _nx = torch.zeros(dim, dtype=torch.long, device=mask.device)
    _ix = torch.zeros_like(_nx)

    x_p = torch.tensor(obj["x_p"], dtype=x[0].dtype, device=x[0].device)
    e_x = torch.tensor(obj["e_x"], dtype=x[0].dtype, device=x[0].device)

    slicer = []
    for i in range(dim):
        x_p[i] = x[i][torch.argmin(abs(x[i] - x_p[i]))]
        _nx[i] = torch.ceil(e_x[i] / dx[i]).type(torch.long) + 1
        _ix[i] = torch.argmin(abs(x[i] - x_p[i]))
        slicer.append(slice(_ix[i], _ix[i] + _nx[i], None))

    mask[slicer] = True

    return mask
