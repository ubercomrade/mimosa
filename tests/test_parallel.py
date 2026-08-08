"""Automatic batch parallelism policy."""

from numba import get_num_threads

from mimosa.parallel import (
    MIN_PARALLEL_ROWS,
    MIN_PARALLEL_ITEMS,
    scan_dispatch_path,
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

    def test_scan_dispatch_uses_rows_for_small_model_groups(self):
        expected = "row-parallel" if get_num_threads() > 1 else "serial"
        assert (
            scan_dispatch_path(
                MIN_PARALLEL_ITEMS,
                rows=MIN_PARALLEL_ROWS,
                groups=2,
            )
            == expected
        )

    def test_scan_dispatch_uses_models_for_large_model_groups(self):
        expected = "model-parallel" if get_num_threads() > 1 else "serial"
        assert (
            scan_dispatch_path(
                MIN_PARALLEL_ITEMS,
                rows=MIN_PARALLEL_ROWS,
                groups=4,
            )
            == expected
        )
