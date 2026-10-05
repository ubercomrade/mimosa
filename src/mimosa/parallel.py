"""Small execution policy for Numba kernels."""

from __future__ import annotations

from numba import get_num_threads

MIN_PARALLEL_ROWS = 64
MIN_PARALLEL_ITEMS = 50_000
MIN_PARALLEL_ALIGNMENT_OPERATIONS = 25_000


def use_parallel(items, *, rows=None):
    """Return whether a workload is large enough to pay for scheduling."""
    if get_num_threads() <= 1:
        return False
    if items < MIN_PARALLEL_ITEMS:
        return False
    if rows is not None and rows < MIN_PARALLEL_ROWS:
        return False
    return True


def alignment_scratch_bytes(rows, shifts):
    """Bytes for the three parallel alignment row-by-shift reductions."""
    return 24 * rows * shifts


def use_alignment_parallel(items, *, rows, shifts, anchor_count):
    """Choose row-parallel alignment from work shape, not score count alone."""
    if not use_parallel(items, rows=rows):
        return False
    # Every shift visits each query and target anchor once.
    return anchor_count * shifts >= MIN_PARALLEL_ALIGNMENT_OPERATIONS
