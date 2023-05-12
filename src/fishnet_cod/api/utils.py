from itertools import chain
from typing import TypeVar

T = TypeVar("T")


def flatten(iterable):
    """Flatten one level of nesting"""
    return chain.from_iterable(iterable)
