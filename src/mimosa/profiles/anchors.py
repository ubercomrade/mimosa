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


def build_anchor_csr(rows, positions, n_rows):
    rows = np.asarray(rows, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative.")
    if rows.size != positions.size:
        raise ValueError("rows and positions must have equal lengths.")
    if rows.size and (np.any(rows < 0) or np.any(rows >= n_rows)):
        raise ValueError("anchor rows must be within 0:n_rows.")
    if positions.size and np.any(positions < 0):
        raise ValueError("anchor positions must be non-negative.")
    if rows.size == 0:
        return AnchorCSR(np.array([], dtype=np.int64), np.zeros(n_rows + 1, dtype=np.int64))
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    sorted_positions = positions[order]
    counts = np.bincount(sorted_rows, minlength=n_rows)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return AnchorCSR(sorted_positions, offsets)


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
