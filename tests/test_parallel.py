"""Automatic batch parallelism policy."""

from numba import get_num_threads

from mimosa.parallel import (
    MIN_PARALLEL_ALIGNMENT_OPERATIONS,
    MIN_PARALLEL_ROWS,
    MIN_PARALLEL_ITEMS,
    alignment_scratch_bytes,
    use_alignment_parallel,
    use_parallel,
)


class TestParallel:
    def test_dispatch_boundaries_stay_serial_below_64_rows_or_50k_items(
        self, monkeypatch
    ):
        monkeypatch.setattr("mimosa.parallel.get_num_threads", lambda: 2)
        assert not use_parallel(49_999, rows=64)
        assert not use_parallel(50_000, rows=63)
        assert use_parallel(50_000, rows=64)

    def test_large_workloads_are_parallel_eligible(self):
        assert use_parallel(
            MIN_PARALLEL_ITEMS, rows=MIN_PARALLEL_ROWS
        ) is (get_num_threads() > 1)

    def test_alignment_scratch_is_three_row_shift_reductions(self):
        assert alignment_scratch_bytes(64, 21) == 24 * 64 * 21

    def test_alignment_dispatch_considers_shape_and_anchor_density(self, monkeypatch):
        monkeypatch.setattr("mimosa.parallel.get_num_threads", lambda: 2)
        assert not use_alignment_parallel(
            MIN_PARALLEL_ITEMS, rows=64, shifts=1, window_radius=0, anchor_count=64
        )
        assert use_alignment_parallel(
            MIN_PARALLEL_ITEMS,
            rows=64,
            shifts=21,
            window_radius=10,
            anchor_count=64,
        )
        assert use_alignment_parallel(
            MIN_PARALLEL_ITEMS,
            rows=64,
            shifts=2,
            window_radius=0,
            anchor_count=MIN_PARALLEL_ALIGNMENT_OPERATIONS,
        )
