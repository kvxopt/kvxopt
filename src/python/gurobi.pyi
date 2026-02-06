from typing import Literal

from ._types import OptionValue
from .base import matrix, spmatrix

GurobiStatus = Literal["optimal", "infeasible", "unbounded", "unknown"]

options: dict[str, OptionValue]


def solve(
    q: matrix | None = ...,
    G_l: matrix | None = ...,
    G: spmatrix | None = ...,
    G_u: matrix | None = ...,
    A: spmatrix | None = ...,
    b: matrix | None = ...,
    P: spmatrix | None = ...,
    x_l: matrix | None = ...,
    x_u: matrix | None = ...,
    options: dict[str, OptionValue] | None = ...,
) -> tuple[GurobiStatus, matrix, matrix]: ...

def qp(
    q: matrix,
    G: spmatrix,
    h: matrix,
    A: spmatrix | None = ...,
    b: matrix | None = ...,
    P: spmatrix | None = ...,
    options: dict[str, OptionValue] | None = ...,
) -> tuple[GurobiStatus, matrix, matrix, matrix]: ...
