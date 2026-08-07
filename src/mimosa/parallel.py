"""Small execution policy for Numba batch kernels."""

from __future__ import annotations

import os

from numba import get_num_threads

MIN_PARALLEL_ROWS = 64
MIN_PARALLEL_ITEMS = 50_000

MIN_PARALLEL_TARGETS = 64
# ponytail: ProcessPool startup ~tens of ms; below this a serial for loop is cheaper.


def use_parallel(items, *, rows=None):
    """Return whether a batch is large enough to pay for Numba scheduling."""
    if get_num_threads() <= 1:
        return False
    if items < MIN_PARALLEL_ITEMS:
        return False
    return rows is None or rows >= MIN_PARALLEL_ROWS


def use_process_pool(n_targets):
    """Return whether compare_many should spread targets across processes."""
    if n_targets < MIN_PARALLEL_TARGETS:
        return False
    return (os.cpu_count() or 1) > 1