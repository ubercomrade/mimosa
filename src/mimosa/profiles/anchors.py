"""Anchor CSR collection: positions with logERR at or above a threshold."""

from __future__ import annotations

import numpy as np

from .._kernels import (
    collect_threshold_anchors_csr,
    count_threshold_anchors_csr,
)

DEFAULT_MIN_LOGERR = 4.0


def parse_min_logerr(value):
    """Return a finite, positive Float32 logERR threshold."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            threshold = np.float32(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_logerr must be a finite, positive Float32 value.") from exc
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("min_logerr must be a finite, positive Float32 value.")
    return threshold


class AnchorCSR:
    __slots__ = ("positions", "offsets")

    def __init__(self, positions, offsets):
        positions = np.ascontiguousarray(positions)
        if positions.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            positions = np.ascontiguousarray(positions, dtype=np.int64)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        if offsets.size == 0:
            raise ValueError("anchor offsets must not be empty.")
        if offsets[0] != 0:
            raise ValueError("anchor offsets must start at 0.")
        if offsets[-1] != positions.size:
            raise ValueError("anchor offsets must end at positions length.")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("anchor offsets must be nondecreasing.")
        self.positions = positions
        self.offsets = offsets

    def __eq__(self, other):
        return (
            isinstance(other, AnchorCSR)
            and np.array_equal(self.positions, other.positions)
            and np.array_equal(self.offsets, other.offsets)
        )

    def __repr__(self):
        return f"AnchorCSR({self.positions.size} anchors, {self.offsets.size - 1} rows)"


def collect_anchor_csr(scores, threshold, *, position_offset=0):
    n = len(scores)
    max_row_length = max(
        (
            int(scores.offsets[row + 1] - scores.offsets[row])
            for row in range(n)
        ),
        default=0,
    )
    max_position = position_offset + max_row_length - 1
    position_dtype = (
        np.int32 if max_position <= np.iinfo(np.int32).max else np.int64
    )
    threshold = parse_min_logerr(threshold)
    offsets = np.empty(n + 1, dtype=np.int64)
    count = count_threshold_anchors_csr(
        scores.data, scores.offsets, threshold, offsets
    )
    positions = np.empty(count, dtype=position_dtype)
    collect_threshold_anchors_csr(
        scores.data, scores.offsets, threshold, positions, offsets
    )
    if position_offset:
        positions += position_offset
    return AnchorCSR(positions, offsets)


def collect_both_anchors(bundle, threshold, *, position_offset=0):
    """Collect anchors in physical site coordinates, not scan-array indices."""
    fwd_csr = collect_anchor_csr(
        bundle.forward, threshold, position_offset=position_offset
    )
    if bundle.forward is bundle.reverse:
        return (fwd_csr, fwd_csr)
    return (
        fwd_csr,
        collect_anchor_csr(
            bundle.reverse, threshold, position_offset=position_offset
        ),
    )
