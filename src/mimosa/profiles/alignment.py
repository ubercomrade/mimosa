"""Profile metrics, shift scoring, and orientation selection."""

from __future__ import annotations

import numpy as np

from .._kernels import _score_shift_best, _score_shift_csr

PROFILE_EPS = 1e-6

ORIENTATION_PAIRS = (("++", 0, 0), ("+-", 0, 1), ("-+", 1, 0), ("--", 1, 1))
ORIENTATION_RANK = {"++": 0, "+-": 1, "-+": 2, "--": 3}


_METRIC_KINDS = {
    "co": (0, False),
    "dice": (0, True),
    "cosine": (1, False),
}


def parse_profile_metric(name):
    name = str(name)
    if name not in _METRIC_KINDS:
        raise ValueError(
            f"profile metric must be one of: {tuple(_METRIC_KINDS)}, got '{name}'."
        )
    return name


def _metric_kind(metric):
    return _METRIC_KINDS[metric][0]


def _use_dice(metric):
    return _METRIC_KINDS[metric][1]


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
    orientation_label,
    search_range,
    window_radius,
    realign_window,
    metric,
    min_logerr=0.0,
):
    query_scores = query_bundle.forward if query_strand == 0 else query_bundle.reverse
    target_scores = target_bundle.forward if target_strand == 0 else target_bundle.reverse
    q_csr = query_anchors[query_strand]
    t_csr = target_anchors[target_strand]

    n_shifts = 2 * search_range + 1
    scores = np.empty(n_shifts, dtype=np.float32)
    site_counts = np.empty(n_shifts, dtype=np.int64)
    out_score = np.empty(1, dtype=np.float64)
    out_sites = np.empty(1, dtype=np.int64)
    kind = _metric_kind(metric)
    use_dice = _use_dice(metric)

    if min_logerr > 0.0:
        max_row_length = (
            int(np.max(np.diff(query_scores.offsets)))
            if query_scores.offsets.size > 1
            else 0
        )
        seen = np.zeros(max_row_length, dtype=np.uint32)
        candidates = np.empty(max_row_length, dtype=np.int64)
        for shift_index in range(n_shifts):
            shift = shift_index - search_range
            _score_shift_csr(
                query_scores.data, query_scores.offsets,
                target_scores.data, target_scores.offsets,
                q_csr.positions, q_csr.offsets,
                t_csr.positions, t_csr.offsets,
                shift, window_radius, realign_window,
                kind, use_dice, seen, candidates, out_score, out_sites,
            )
            scores[shift_index] = out_score[0]
            site_counts[shift_index] = out_sites[0]
    else:
        for shift_index in range(n_shifts):
            shift = shift_index - search_range
            _score_shift_best(
                query_scores.data, query_scores.offsets,
                target_scores.data, target_scores.offsets,
                q_csr.positions, q_csr.offsets,
                t_csr.positions, t_csr.offsets,
                shift, window_radius, realign_window,
                kind, use_dice, out_score, out_sites,
            )
            scores[shift_index] = out_score[0]
            site_counts[shift_index] = out_sites[0]

    best_score = np.float32(0.0)
    best_shift = 0
    best_n_sites = 0
    for shift_index in range(n_shifts):
        shift = shift_index - search_range
        score = scores[shift_index]
        n_sites = site_counts[shift_index]
        if float(score) > float(best_score) or (
            float(score) == float(best_score)
            and (
                n_sites > best_n_sites
                or (n_sites == best_n_sites and abs(shift) < abs(best_shift))
            )
        ):
            best_score = score
            best_shift = shift
            best_n_sites = n_sites
    return best_score, best_shift, best_n_sites, orientation_label


class ProfileConfig:
    def __init__(
        self,
        metric="co",
        search_range=10,
        window_radius=10,
        realign_window=3,
        min_logerr=0.0,
    ):
        if search_range < 0:
            raise ValueError("search_range must be non-negative.")
        if window_radius < 0:
            raise ValueError("window_radius must be non-negative.")
        if realign_window < 0:
            raise ValueError("realign_window must be non-negative.")
        if not np.isfinite(min_logerr):
            raise ValueError("min_logerr must be finite.")
        self.metric = parse_profile_metric(metric)
        self.search_range = int(search_range)
        self.window_radius = int(window_radius)
        self.realign_window = int(realign_window)
        self.min_logerr = np.float32(min_logerr)


def profile_compare(query_bundle, query_anchors, target_bundle, target_anchors, config):
    metric = config.metric
    best_score = np.float32(0.0)
    best_shift = 0
    best_n_sites = 0
    best_orientation = "++"
    best_rank = 2**63 - 1

    for label, q_strand, t_strand in _orientation_pairs(query_bundle, target_bundle):
        score, shift, n_sites, _ = _score_orientation_pair(
            query_bundle,
            target_bundle,
            query_anchors,
            target_anchors,
            q_strand,
            t_strand,
            label,
            config.search_range,
            config.window_radius,
            config.realign_window,
            metric,
            config.min_logerr,
        )
        rank = ORIENTATION_RANK[label]
        if float(score) > float(best_score) or (
            float(score) == float(best_score)
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

    return best_score, best_shift, best_orientation, best_n_sites, parse_profile_metric(metric)
