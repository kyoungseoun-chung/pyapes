#!/usr/bin/env python3
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyABC.core.backend import DType
from pyABC.core.backend import DTYPE_DOUBLE
from pyABC.core.backend import DTYPE_SINGLE
from pyABC.core.backend import TORCH_DEVICE
from pyABC.core.backend import TorchDevice
from pyABC.core.geometry import GeoTypeIdentifier
from pyABC.core.geometry.basis import Geometry


class Mesh:
    """Equidistance rectangular (in mind) mesh."""

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
            self._nx = torch.tensor(
                spacing,
                dtype=self.dtype.int,
                device=self.device.type,
                requires_grad=False,
            )
            self._dx = self._lx / (self._nx.type(self.dtype.float) - 1.0)
        elif float in GeoTypeIdentifier(spacing):
            # Node information
            self._dx = torch.tensor(
                spacing,
                dtype=self.dtype.float,
                device=self.device,
                requires_grad=False,
            )
            self._nx = (self._lx / self._dx + 1).type(self.dtype.int)
        else:
            raise TypeError("Mesh: spacing only accept int or float")

        self.x = []

        for i in range(self.dim):
            self.x.append(
                torch.linspace(
                    self.lower[i].item(),
                    self.upper[i].item(),
                    int(self.nx[i]),
                    dtype=self.dtype.float,
                    device=self.device,
                )
            )

        # Mesh grid
        self.grid = torch.meshgrid(self.x, indexing="ij")

    @property
    def dim(self) -> int:
        return self.domain.dim

    @property
    def X(self) -> Tensor:
        return self.grid[0]

    @property
    def Y(self) -> Tensor:
        """If `self.dim > 1` return empty tensor."""
        return (
            self.grid[1]
            if self.dim > 1
            else torch.tensor([], dtype=self.dtype.float)
        )

    @property
    def Z(self) -> Tensor:
        """If `self.dim > 2` return empty tensor."""
        return (
            self.grid[2]
            if self.dim > 2
            else torch.tensor([], dtype=self.dtype.float)
        )

    @property
    def N(self) -> int:
        """Return total number of grid points."""

        return int(torch.prod(self._nx, dtype=self.dtype.int))

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
        return self._dx

    @property
    def nx(self) -> Tensor:
        return self._nx

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


def field_patch_mask(mesh: Mesh) -> tuple[dict, dict]:
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

    domain = mesh.domain.config
    obstacle = mesh.obstacle

    dtype = mesh.dtype
    device = mesh.device

    mask_to_save = dict()
    obj_mask_sep = dict()

    # Loop over patch objects
    for obj in domain:

        mask = torch.zeros(*nx, dtype=dtype.bool, device=device)
        mask = get_patch_mask(x, dx, obj, mask, dim)

        # Save as sub dictionary
        # Should id be face dir?
        mask_to_save["Domain-" + obj["name"]] = mask

    if obstacle is not None:

        raise NotImplementedError(
            "field_path_mask: inner obstacle is not supported yet!"
        )

    return mask_to_save, obj_mask_sep


def get_patch_mask(
    x: list[Tensor],
    dx: Tensor,
    obj: dict,
    mask: Tensor,
    dim: int,
) -> Tensor:
    """Get patch mask."""

    _nx = torch.zeros(dim, dtype=torch.long)
    _ix = torch.zeros_like(_nx)

    x_p = obj["geometry"]["x_p"]
    e_x = obj["geometry"]["e_x"]

    for i in range(dim):
        x_p[i] = x[i][torch.argmin(abs(x[i] - x_p[i]))]
        _nx[i] = torch.ceil(e_x[i] / dx[i]).type(torch.long) + 1
        _ix[i] = torch.argmin(abs(x[i] - x_p[i]))

    mask = _assign_mask(mask, _ix, _nx, dim)

    return mask


def _assign_mask(mask: Tensor, ix: Tensor, nx: Tensor, dim: int) -> Tensor:

    if dim == 1:
        mask[
            ix[0] : ix[0] + nx[0],
        ] = True
    elif dim == 2:
        mask[
            ix[0] : ix[0] + nx[0],
            ix[1] : ix[1] + nx[1],
        ] = True
    else:
        mask[
            ix[0] : ix[0] + nx[0],
            ix[1] : ix[1] + nx[1],
            ix[2] : ix[2] + nx[2],
        ] = True

    return mask
