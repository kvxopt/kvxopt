from ._types import OptionValue
from .base import matrix, spmatrix

options: dict[str, OptionValue]


def order(A: spmatrix) -> matrix: ...
