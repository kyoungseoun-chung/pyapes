#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Union

import torch

TORCH_DEVICE = ["cpu", "cuda", "mps"]
"""Available `torch` device. Supports `cpu` and `cuda`. `mps` is experimental (lack of too many core functions at this point)."""
DTYPE_SINGLE = ["single", "s", 32]
DTYPE_DOUBLE = ["double", "d", 64]


@dataclass
class DType:
    """Defining data type for the simulation.

    Examples:
        >>> dtype = DType("single")     # data type as single
        >>> dtype.float
        torch.float23
        >>> dtype = DType("double")     # data type as double
        >>> dtype.float
        torch.float64
    """

    precision: Union[str, int] = "double"

    def __post_init__(self):
        if self.precision in DTYPE_SINGLE:
            # Set default precision to single
            torch.set_default_tensor_type(torch.FloatTensor)
            self._float = torch.float32
            self._complex = torch.complex64
            self._int = torch.int32
            self._bool = torch.bool
        elif self.precision in DTYPE_DOUBLE:
            # Set default precision to double
            torch.set_default_tensor_type(torch.DoubleTensor)
            self._float = torch.float64
            self._complex = torch.complex128
            self._int = torch.int64
            self._bool = torch.bool
        else:
            raise ValueError("Invalid precision type!")

    @property
    def float(self) -> torch.dtype:
        """Return float type."""
        return self._float

    @property
    def int(self) -> torch.dtype:
        """Return int type."""
        return self._int

    @property
    def complex(self) -> torch.dtype:
        """Return complex type."""
        return self._complex

    @property
    def bool(self) -> torch.dtype:
        """Return boolean type."""
        return self._bool

    def __repr__(self):
        return f"(torch.dtype){self.precision}"


@dataclass
class TorchDevice:
    """Compute device.

    Examples:
        >>> device = TorchDevice("cpu")
        >>> device.type
        device(type='cpu')

    """

    device_type: str = "cpu"

    def __init__(self, device_type: str = "cpu"):
        assert device_type in TORCH_DEVICE

        self._device = torch.device(device_type.lower())

    @property
    def device(self) -> torch.device:
        """Return torch device type."""
        return self._device

    def __repr__(self) -> str:
        return f"Device on {self.device}"
