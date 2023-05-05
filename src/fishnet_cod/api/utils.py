from itertools import chain
from typing import Iterable, TypeVar

T = TypeVar("T")


def unique(iterable: Iterable[T]) -> Iterable[T]:
    """Return unique items from iterable, preserving order."""
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item


def flatten(iterable):
    """Flatten one level of nesting"""
    return chain.from_iterable(iterable)
