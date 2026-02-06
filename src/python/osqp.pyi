from typing import Literal

from ._types import OptionValue
from .base import matrix, spmatrix

OSQPStatus = Literal[
    "solved",
    "solved inaccurate",
    "primal infeasible",
    "primal infeasible inaccurate",
    "dual infeasible",
    "dual infeasible inaccurate",
    "maximum iterations reached",
    "run time limit reached",
    "problem non convex",
    "interrupted by user",
    "unsolved",
]

options: dict[str, OptionValue]


def solve(
    q: matrix,
    A: spmatrix,
    l: matrix,
    u: matrix,
    P: spmatrix | None = ...,
    options: dict[str, OptionValue] | None = ...,
) -> tuple[OSQPStatus, matrix, matrix]: ...

def qp(
    q: matrix,
    G: spmatrix,
    h: matrix,
    A: spmatrix | None = ...,
    b: matrix | None = ...,
    P: spmatrix | None = ...,
    options: dict[str, OptionValue] | None = ...,
) -> tuple[OSQPStatus, matrix, matrix, matrix]: ...
