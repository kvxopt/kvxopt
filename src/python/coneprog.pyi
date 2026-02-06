from typing import TypeAlias

from ._types import OptionValue, SolverStatus
from .base import matrix, spmatrix

options: dict[str, OptionValue]

ConeArg: TypeAlias = (
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


def conelp(c: matrix, G: matrix | spmatrix, h: matrix, dims: dict[str, OptionValue] | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., primalstart: dict[str, matrix] | None = ..., dualstart: dict[str, matrix | list[matrix]] | None = ..., kktsolver: str | None = ..., xnewcopy: str | None = ..., xdot: str | None = ..., xaxpy: str | None = ..., xscal: str | None = ..., ynewcopy: str | None = ..., ydot: str | None = ..., yaxpy: str | None = ..., yscal: str | None = ..., **kwargs: ConeArg) -> SolverResult: ...
def coneqp(P: matrix | spmatrix, q: matrix, G: matrix | spmatrix | None = ..., h: matrix | None = ..., dims: dict[str, OptionValue] | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., initvals: dict[str, matrix | list[matrix]] | None = ..., kktsolver: str | None = ..., xnewcopy: str | None = ..., xdot: str | None = ..., xaxpy: str | None = ..., xscal: str | None = ..., ynewcopy: str | None = ..., ydot: str | None = ..., yaxpy: str | None = ..., yscal: str | None = ..., **kwargs: ConeArg) -> SolverResult: ...
def lp(c: matrix, G: matrix | spmatrix, h: matrix, A: matrix | spmatrix | None = ..., b: matrix | None = ..., kktsolver: str | None = ..., solver: str | None = ..., primalstart: dict[str, matrix] | None = ..., dualstart: dict[str, matrix | list[matrix]] | None = ..., **kwargs: ConeArg) -> SolverResult: ...
def socp(c: matrix, Gl: matrix | spmatrix | None = ..., hl: matrix | None = ..., Gq: list[matrix | spmatrix] | None = ..., hq: list[matrix] | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., kktsolver: str | None = ..., solver: str | None = ..., primalstart: dict[str, matrix] | None = ..., dualstart: dict[str, matrix | list[matrix]] | None = ..., **kwargs: ConeArg) -> SolverResult: ...
def sdp(c: matrix, Gl: matrix | spmatrix | None = ..., hl: matrix | None = ..., Gs: list[matrix | spmatrix] | None = ..., hs: list[matrix] | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., kktsolver: str | None = ..., solver: str | None = ..., primalstart: dict[str, matrix] | None = ..., dualstart: dict[str, matrix | list[matrix]] | None = ..., **kwargs: ConeArg) -> SolverResult: ...
def qp(P: matrix | spmatrix, q: matrix, G: matrix | spmatrix | None = ..., h: matrix | None = ..., A: matrix | spmatrix | None = ..., b: matrix | None = ..., solver: str | None = ..., kktsolver: str | None = ..., initvals: dict[str, matrix | list[matrix]] | None = ..., **kwargs: ConeArg) -> SolverResult: ...
