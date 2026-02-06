from ._types import Number, OptionValue
from .base import matrix, spmatrix

options: dict[str, OptionValue]


def __getattr__(name: str) -> matrix | spmatrix | Number | dict[str, OptionValue]: ...
