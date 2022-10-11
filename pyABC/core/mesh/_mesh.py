#!/usr/bin/env python3
from typing import Optional, Union

import torch
from torch import Tensor

from pyABC.core.geometry import Geometry, GeoTypeIdentifier
from pyABC.core.backend import (
    DType,
    TorchDevice,
    DTYPE_DOUBLE,
    DTYPE_SINGLE,
    TORCH_DEVICE,
)


class Mesh:
    """Equidistance rectangular (in mind) mesh."""

    def __init__(
        self,
        domain: Geometry,
        obstacle: Optional[list[Geometry]],
        spacing: list[Union[int, float]] = [],
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
    def nx(self) -> list[int]:
        return self._nx.tolist()

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
