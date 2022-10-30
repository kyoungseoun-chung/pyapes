#!/usr/bin/env python3
"""Finite volume discretizer base class to be used in `pyapes.core.solver.fvc` and `pyapes.core.solver.fvm`"""
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pyapes.core.variables import Field
from pyapes.core.variables import Flux


@dataclass(eq=False)
class Discretizer:
    """Base class of FVM discretization."""

    # Init relevant attributes
    _ops: dict[int, dict[str, Union[Callable, str]]] = field(
        default_factory=dict
    )
    _rhs: Optional[Tensor] = None

    @property
    def ops(self) -> dict[int, dict[str, Union[Callable, str]]]:
        """Collection of operators used in `pyapes.core.solver.Solver().set_eq()`"""
        return self._ops

    @property
    def rhs(self) -> Optional[Tensor]:
        """RHS of `set_eq()`"""
        return self._rhs

    @property
    def var(self) -> Field:
        """Primary Field variable to be discretized."""
        raise NotImplementedError

    @property
    def flux(self) -> Flux:
        """Flux object to be computed."""
        raise NotImplementedError

    def __eq__(self, other: Union[Tensor, float]) -> Any:

        if isinstance(other, Tensor):
            self._rhs = other
        else:
            self._rhs = torch.zeros_like(self.var()) + other

        return self

    def __add__(self, other: Any) -> Any:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update(
            {idx[-1] + 1: {"flux": other.flux, "op": other.__class__.__name__}}
        )

        return self

    def __sub__(self, other: Any) -> Any:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update(
            {
                idx[-1]
                + 1: {"flux": other.flux * -1, "op": other.__class__.__name__}
            }
        )

        return self
