"""Weighted Jaccard scoring over circular shifts and orientation selection."""

from __future__ import annotations

import numpy as np

from ..models import strict_integer
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
    offsets = query_scores.offsets
    if offsets.size != target_scores.offsets.size:
        raise ValueError("profiles must have the same number of rows.")
    if not np.array_equal(offsets, target_scores.offsets):
        raise ValueError("profiles must have identical row lengths.")
    import jax
    from .._jax_backend import compare_circular

    q = np.stack((query_scores.data, query_scores.data))
    t = np.stack((target_scores.data, target_scores.data))
    with jax.enable_x64(True):
        score, shift, _, sites = compare_circular(
            q, t, offsets, min_logerr=min_logerr, search_range=search_range
        )
        return float(score), int(shift), int(sites)


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
    import jax
    from .._jax_backend import compare_circular

    offsets = query_bundle.forward.offsets
    if offsets.size != target_bundle.forward.offsets.size:
        raise ValueError("profiles must have the same number of rows.")
    if not np.array_equal(offsets, target_bundle.forward.offsets):
        raise ValueError("profiles must have identical row lengths.")
    query = np.stack((query_bundle.forward.data, query_bundle.reverse.data))
    target = np.stack((target_bundle.forward.data, target_bundle.reverse.data))
    with jax.enable_x64(True):
        score, shift, orientation, sites = compare_circular(
            query, target, offsets, min_logerr=config.min_logerr,
            search_range=config.search_range,
        )
        return np.float32(score), int(shift), ORIENTATION_PAIRS[int(orientation)][0], int(sites)
