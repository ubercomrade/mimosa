"""Small execution policy for Numba batch kernels."""

from __future__ import annotations

from numba import get_num_threads

MIN_PARALLEL_ROWS = 64
MIN_PARALLEL_ITEMS = 50_000
MIN_PARALLEL_TARGETS = 64
MIN_PARALLEL_MODEL_GROUPS = 4


def use_parallel(items, *, rows=None, groups=None):
    """Return whether a batch is large enough to pay for Numba scheduling."""
    if get_num_threads() <= 1:
        return False
    if items < MIN_PARALLEL_ITEMS:
        return False
    if rows is not None and rows < MIN_PARALLEL_ROWS:
        return False
    return groups is None or groups >= MIN_PARALLEL_MODEL_GROUPS


def dispatch_path(items, *, rows=None, groups=None):
    """Describe the Numba dispatch selected for a batch."""
    if not use_parallel(items, rows=rows, groups=groups):
        return "serial"
    return "model-parallel" if groups is not None else "row-parallel"


def scan_dispatch_path(items, *, rows, groups):
    """Select model-parallel, row-parallel, or serial scan execution."""
    if use_parallel(items, rows=rows, groups=groups):
        return "model-parallel"
    if use_parallel(items, rows=rows):
        return "row-parallel"
    return "serial"
