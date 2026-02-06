from typing import Literal

from ._types import OptionValue
from .base import matrix, spmatrix

DSDPStatus = Literal["DSDP_PDFEASIBLE", "DSDP_UNBOUNDED", "DSDP_INFEASIBLE", "DSDP_UNKNOWN"]

options: dict[str, OptionValue]


def sdp(
    c: matrix,
    Gl: matrix | spmatrix | None = ...,
    hl: matrix | None = ...,
    Gs: list[matrix | spmatrix] | None = ...,
    hs: list[matrix] | None = ...,
    gamma: float = ...,
    beta: float = ...,
    options: dict[str, OptionValue] | None = ...,
) -> tuple[DSDPStatus, matrix, float, matrix, list[matrix]]: ...
