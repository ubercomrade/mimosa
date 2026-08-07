"""Small execution policy for Numba batch kernels."""

from __future__ import annotations

from numba import get_num_threads

MIN_PARALLEL_ROWS = 64
MIN_PARALLEL_ITEMS = 50_000


def use_parallel(items, *, rows=None):
    """Return whether a batch is large enough to pay for Numba scheduling."""
    if get_num_threads() <= 1:
        return False
    if items < MIN_PARALLEL_ITEMS:
        return False
    return rows is None or rows >= MIN_PARALLEL_ROWS
