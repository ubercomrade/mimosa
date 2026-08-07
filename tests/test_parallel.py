"""Automatic batch parallelism policy."""

import os

from numba import get_num_threads

from mimosa.parallel import (
    MIN_PARALLEL_ITEMS,
    MIN_PARALLEL_ROWS,
    MIN_PARALLEL_TARGETS,
    use_parallel,
    use_process_pool,
)


class TestParallel:
    def test_small_workloads_stay_serial(self):
        assert not use_parallel(MIN_PARALLEL_ITEMS - 1, rows=MIN_PARALLEL_ROWS)
        assert not use_parallel(MIN_PARALLEL_ITEMS, rows=MIN_PARALLEL_ROWS - 1)

    def test_large_workloads_are_parallel_eligible(self):
        assert use_parallel(
            MIN_PARALLEL_ITEMS, rows=MIN_PARALLEL_ROWS
        ) is (get_num_threads() > 1)


class TestProcessPool:
    def test_small_batches_stay_serial(self):
        assert not use_process_pool(MIN_PARALLEL_TARGETS - 1)

    def test_large_batches_are_process_eligible(self):
        assert use_process_pool(MIN_PARALLEL_TARGETS) is ((os.cpu_count() or 1) > 1)
