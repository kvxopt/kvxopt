from typing import Literal, overload

from ._types import OptionValue
from .base import matrix, spmatrix

GLPKStatus = Literal["optimal", "primal infeasible", "dual infeasible", "unknown"]

options: dict[str, OptionValue]


@overload
def lp(c: matrix, G: matrix | spmatrix, h: matrix, A: None = ..., b: None = ..., options: dict[str, OptionValue] | None = ...) -> tuple[GLPKStatus, matrix | None, matrix | None]: ...
@overload
def lp(c: matrix, G: matrix | spmatrix, h: matrix, A: matrix | spmatrix, b: matrix, options: dict[str, OptionValue] | None = ...) -> tuple[GLPKStatus, matrix | None, matrix | None, matrix | None]: ...

def ilp(
    c: matrix,
    G: matrix | spmatrix,
    h: matrix,
    A: matrix | spmatrix | None,
    b: matrix | None,
    I: set[int],
    B: set[int],
    options: dict[str, OptionValue] | None = ...,
) -> tuple[GLPKStatus, matrix | None]: ...
