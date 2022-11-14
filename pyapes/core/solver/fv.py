#!/usr/bin/env python3
"""Finite volume discretizer base class to be used in `pyapes.core.solver.fvc` and `pyapes.core.solver.fvm`"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import TypedDict

import torch
from torch import Tensor

from pyapes.core.variables import Field
from pyapes.core.variables import Flux


class OPStype(TypedDict):
    """Typed dict for the operation types."""

    name: str
    Aop: Callable[..., Tensor]
    inputs: tuple[float, Field] | tuple[Field, Field] | tuple[Field]
    var: Field
    sign: float | int


@dataclass(eq=False)
class Discretizer:
    """Base class of FVM discretization."""

    # Init relevant attributes
    _ops: dict[int, OPStype] = field(default_factory=dict)
    _rhs: Tensor | None = None

    @property
    def ops(self) -> dict[int, OPStype]:
        """Collection of operators used in `pyapes.core.solver.Solver().set_eq()`"""
        return self._ops

    @property
    def rhs(self) -> Tensor | None:
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

    def __eq__(self, other: Tensor | float) -> Discretizer:

        if isinstance(other, Tensor):
            self._rhs = other
        else:
            self._rhs = torch.zeros_like(self.var()) + other

        return self

    def __add__(self, other: Discretizer) -> Discretizer:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        self._ops.update({idx[-1] + 1: other.ops[0]})

        return self

    def __sub__(self, other: Discretizer) -> Discretizer:

        assert self.flux is not None, "Discretizer: Flux is not assigned!"

        idx = list(self._ops.keys())
        other.ops[0]["sign"] = -1
        self._ops.update({idx[-1] + 1: other.ops[0]})

        return self
