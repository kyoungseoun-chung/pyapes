#!/usr/bin/env python3
"""Collection of types used in solver module."""
from typing import Callable
from typing import TypedDict

from torch import Tensor

from pyapes.variables import Field


class DivConfigType(TypedDict):
    limiter: str
    edge: bool


class LaplacianConfigType(TypedDict):
    limiter: str
    edge: bool


class GradConfigType(TypedDict):
    limiter: str
    edge: bool


class DdtConfigType(TypedDict):
    scheme: str


class DiscretizerConfigType(TypedDict, total=False):
    div: DivConfigType
    laplacian: LaplacianConfigType
    grad: GradConfigType
    ddt: DdtConfigType


GEN_RHS = Callable[[Field], Tensor]
DIV_RHS = Callable[[Field | Tensor | float, Field, DiscretizerConfigType], Tensor]


class OPStype(TypedDict):
    """Typed dict for the operation types."""

    name: str
    """Operator names"""
    Aop: Callable[
        [Tensor | float | None, Field, list[list[Tensor]]], Tensor
    ] | Callable[
        [Field | Tensor | float, DiscretizerConfigType, Field, list[list[Tensor]]],
        Tensor,
    ]
    """Linear system operator. `Aop` is equivalent to `Ax` in `Ax = b`."""
    target: Field
    """Target field to be discretized."""
    param: tuple[float | Tensor | None, ...] | tuple[
        Field | Tensor | float, DiscretizerConfigType
    ]
    """Additional parameters other than target. e.g. `coeff` in `laplacian(coeff, var)`."""
    sign: float | int
    """Sign to be applied."""
    other: dict[str, float] | None
    """Additional information. e.g. `dt` in `Ddt`."""
    A_coeffs: list[list[Tensor]]
    """Coefficients of the discretization."""
    adjust_rhs: GEN_RHS | DIV_RHS
    # adjust_rhs: Any
    """Tensor used to adjust rhs."""
