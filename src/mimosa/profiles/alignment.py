"""Weighted Jaccard scoring over circular shifts and orientation selection."""

from __future__ import annotations

import numpy as np

from ..models import strict_integer
from .._kernels import _jaccard_orientation, _jaccard_orientation_parallel
from ..parallel import use_alignment_parallel
from .anchors import DEFAULT_MIN_LOGERR, parse_min_logerr

METRIC_NAME = "jaccard"
ORIENTATION_PAIRS = (("++", 0, 0), ("+-", 0, 1), ("-+", 1, 0), ("--", 1, 1))
ORIENTATION_RANK = {"++": 0, "+-": 1, "-+": 2, "--": 3}


def _orientation_pairs(query_bundle, target_bundle):
    qs = query_bundle.forward is query_bundle.reverse
    ts = target_bundle.forward is target_bundle.reverse
    if qs and ts:
        return (("++", 0, 0),)
    if qs:
        return (("++", 0, 0), ("+-", 0, 1))
    if ts:
        return (("++", 0, 0), ("-+", 1, 0))
    return ORIENTATION_PAIRS


def _score_orientation_pair(
    query_bundle,
    target_bundle,
    query_anchors,
    target_anchors,
    query_strand,
    target_strand,
    search_range,
    min_logerr,
):
    query_scores = query_bundle.forward if query_strand == 0 else query_bundle.reverse
    target_scores = target_bundle.forward if target_strand == 0 else target_bundle.reverse
    q_csr = query_anchors[query_strand]
    t_csr = target_anchors[target_strand]

    offsets = query_scores.offsets
    if offsets.size != target_scores.offsets.size:
        raise ValueError("profiles must have the same number of rows.")
    if not np.array_equal(offsets, target_scores.offsets):
        raise ValueError("profiles must have identical row lengths.")
    n_rows = offsets.size - 1
    n_shifts = 2 * search_range + 1
    out_scores = np.empty(n_shifts, dtype=np.float64)
    out_sites = np.empty(n_shifts, dtype=np.int64)
    kernel_args = (
        query_scores.data, target_scores.data, offsets,
        q_csr.positions, q_csr.offsets,
        t_csr.positions, t_csr.offsets,
        search_range, np.float32(min_logerr),
    )
    if use_alignment_parallel(
        int(query_scores.data.size + target_scores.data.size),
        rows=n_rows,
        shifts=n_shifts,
        anchor_count=q_csr.positions.size + t_csr.positions.size,
    ):
        _jaccard_orientation_parallel(
            *kernel_args,
            np.empty((n_rows, n_shifts), dtype=np.float64),
            np.empty((n_rows, n_shifts), dtype=np.float64),
            np.empty((n_rows, n_shifts), dtype=np.int64),
            out_scores, out_sites,
        )
    else:
        _jaccard_orientation(*kernel_args, out_scores, out_sites)

    best_score = 0.0
    best_shift = 0
    best_n_sites = 0
    for shift_index in range(n_shifts):
        shift = shift_index - search_range
        score = float(out_scores[shift_index])
        n_sites = int(out_sites[shift_index])
        if score > best_score or (
            score == best_score
            and (
                n_sites > best_n_sites
                or (n_sites == best_n_sites and abs(shift) < abs(best_shift))
            )
        ):
            best_score = score
            best_shift = shift
            best_n_sites = n_sites
    return best_score, best_shift, best_n_sites


class ProfileConfig:
    def __init__(self, search_range=5, min_logerr=DEFAULT_MIN_LOGERR):
        search_range = strict_integer(search_range, "search_range")
        if search_range < 0:
            raise ValueError("search_range must be non-negative.")
        self.search_range = search_range
        self.min_logerr = parse_min_logerr(min_logerr)


def profile_compare(
    query_bundle,
    query_anchors,
    target_bundle,
    target_anchors,
    config,
):
    """Return the best ``(score, shift, orientation, n_sites)`` alignment.

    Bundles must be circular logERR profiles with identical row layouts, and
    anchors must list the positions at or above ``config.min_logerr``.
    """
    best_score = 0.0
    best_shift = 0
    best_n_sites = 0
    best_orientation = "++"
    best_rank = 2**63 - 1

    for label, q_strand, t_strand in _orientation_pairs(query_bundle, target_bundle):
        score, shift, n_sites = _score_orientation_pair(
            query_bundle,
            target_bundle,
            query_anchors,
            target_anchors,
            q_strand,
            t_strand,
            config.search_range,
            config.min_logerr,
        )
        rank = ORIENTATION_RANK[label]
        if score > best_score or (
            score == best_score
            and (
                n_sites > best_n_sites
                or (
                    n_sites == best_n_sites
                    and (
                        abs(shift) < abs(best_shift)
                        or (abs(shift) == abs(best_shift) and rank < best_rank)
                    )
                )
            )
        ):
            best_score = score
            best_shift = shift
            best_n_sites = n_sites
            best_orientation = label
            best_rank = rank

    return np.float32(best_score), best_shift, best_orientation, best_n_sites
