from typing import TypeAlias

from ._types import OptionValue, SolverStatus
from .base import matrix, spmatrix

SolverValue: TypeAlias = (
    SolverStatus
    | matrix
    | spmatrix
    | list[matrix]
    | list[spmatrix]
    | dict[str, OptionValue]
    | float
    | int
    | None
)
SolverResult: TypeAlias = dict[str, SolverValue]

options: dict[str, OptionValue]


def conelp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def coneqp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def lp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def socp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def sdp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def qp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def cp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def cpl(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
def gp(*args: matrix | spmatrix | float | int | str, **kwargs: matrix | spmatrix | float | int | str | dict[str, OptionValue]) -> SolverResult: ...
