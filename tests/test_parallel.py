"""Automatic batch parallelism policy."""

from numba import get_num_threads

from mimosa.parallel import (
    MIN_PARALLEL_ROWS,
    MIN_PARALLEL_ITEMS,
    use_parallel,
)


class TestParallel:
    def test_small_workloads_stay_serial(self):
        assert not use_parallel(MIN_PARALLEL_ITEMS - 1, rows=MIN_PARALLEL_ROWS)
        assert not use_parallel(MIN_PARALLEL_ITEMS, rows=MIN_PARALLEL_ROWS - 1)

    def test_large_workloads_are_parallel_eligible(self):
        assert use_parallel(
            MIN_PARALLEL_ITEMS, rows=MIN_PARALLEL_ROWS
        ) is (get_num_threads() > 1)
