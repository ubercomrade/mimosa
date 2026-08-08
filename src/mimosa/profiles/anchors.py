"""Anchor CSR collection: best-per-row or threshold-based."""

from __future__ import annotations

import numpy as np

from .._kernels import (
    collect_best_anchors_csr,
    collect_threshold_anchors_csr,
)


class AnchorCSR:
    __slots__ = ("positions", "offsets")

    def __init__(self, positions, offsets):
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


def collect_anchor_csr(scores, threshold):
    n = len(scores)
    positions = np.empty(
        scores.data.size if threshold > 0.0 else n,
        dtype=np.int64,
    )
    offsets = np.empty(n + 1, dtype=np.int64)
    if threshold > 0.0:
        count = collect_threshold_anchors_csr(
            scores.data,
            scores.offsets,
            np.float32(threshold),
            positions,
            offsets,
        )
    else:
        count = collect_best_anchors_csr(scores.data, scores.offsets, positions, offsets)
    return AnchorCSR(positions[:count], offsets)


def collect_both_anchors(bundle, threshold):
    fwd_csr = collect_anchor_csr(bundle.forward, threshold)
    if bundle.forward is bundle.reverse:
        return (fwd_csr, fwd_csr)
    return (fwd_csr, collect_anchor_csr(bundle.reverse, threshold))
