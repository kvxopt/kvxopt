from typing import Literal, Mapping, Protocol, Sequence, TypeAlias

Number: TypeAlias = int | float | complex
RealNumber: TypeAlias = int | float
Scalar: TypeAlias = Number | bool | str

Trans: TypeAlias = Literal["N", "T", "C"]
Uplo: TypeAlias = Literal["L", "U"]
Side: TypeAlias = Literal["L", "R"]
Diag: TypeAlias = Literal["N", "U"]

OptionValue: TypeAlias = (
    Scalar
    | None
    | Sequence["OptionValue"]
    | Mapping[str, "OptionValue"]
)

SolverStatus: TypeAlias = Literal[
    "optimal",
    "unknown",
    "primal infeasible",
    "dual infeasible",
    "solved",
    "solved inaccurate",
    "primal infeasible inaccurate",
    "dual infeasible inaccurate",
    "DSDP_PDFEASIBLE",
    "DSDP_UNBOUNDED",
    "DSDP_INFEASIBLE",
    "DSDP_UNKNOWN",
]


class FactorHandle(Protocol):
    pass
