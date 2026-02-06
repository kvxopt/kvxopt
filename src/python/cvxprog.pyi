from typing import TypeAlias

from ._types import OptionValue, SolverStatus
from .base import matrix, spmatrix

options: dict[str, OptionValue]

CvxArg: TypeAlias = (
    matrix
    | spmatrix
    | list[matrix]
    | list[spmatrix]
    | tuple[matrix, ...]
    | tuple[spmatrix, ...]
    | dict[str, OptionValue]
    | float
    | int
    | str
    | None
)
SolverValue: TypeAlias = SolverStatus | matrix | spmatrix | list[matrix] | float | int | dict[str, OptionValue] | None
SolverResult: TypeAlias = dict[str, SolverValue]


def cpl(c: matrix, F: CvxArg, G: matrix | spmatrix | None = ..., h: matrix | None = ..., dims: dict[str, OptionValue] | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., kktsolver: str | None = ..., xnewcopy: str | None = ..., xdot: str | None = ..., xaxpy: str | None = ..., xscal: str | None = ..., ynewcopy: str | None = ..., ydot: str | None = ..., yaxpy: str | None = ..., yscal: str | None = ..., **kwargs: CvxArg) -> SolverResult: ...
def cp(F: CvxArg, G: matrix | spmatrix | None = ..., h: matrix | None = ..., dims: dict[str, OptionValue] | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., kktsolver: str | None = ..., xnewcopy: str | None = ..., xdot: str | None = ..., xaxpy: str | None = ..., xscal: str | None = ..., ynewcopy: str | None = ..., ydot: str | None = ..., yaxpy: str | None = ..., yscal: str | None = ..., **kwargs: CvxArg) -> SolverResult: ...
def gp(K: matrix, F: matrix | spmatrix, g: matrix, G: matrix | spmatrix | None = ..., h: matrix | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., kktsolver: str | None = ..., **kwargs: CvxArg) -> SolverResult: ...
