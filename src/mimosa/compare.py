"""Comparison orchestration: prepare profiles and compare."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MotifModel
from .profiles.alignment import ProfileConfig, parse_profile_metric, profile_compare
from .profiles.prepared import PreparedProfile, ScoreProfile, prepare_profile


@dataclass(frozen=True)
class ComparisonResult:
    query: str
    target: str
    score: np.float32
    offset: int
    orientation: str
    metric: str
    n_sites: int = 0

    def to_dict(self):
        d = {
            "query": self.query,
            "target": self.target,
            "score": float(self.score),
            "offset": self.offset,
            "orientation": self.orientation,
            "metric": self.metric,
        }
        if self.n_sites > 0:
            d["n_sites"] = int(self.n_sites)
        return d


def _check_threshold(threshold, prepared):
    if threshold != prepared.min_logerr:
        raise ValueError("min_logerr differs from the prepared query threshold.")


def _prepare_side(model, sequences, background, threshold, normalization, cache):
    if isinstance(model, PreparedProfile):
        return model
    if isinstance(model, MotifModel):
        if sequences is None:
            raise ValueError("motif comparison requires comparison sequences.")
        return prepare_profile(model, sequences, background=background, min_logerr=threshold, normalization=normalization, cache=cache)
    if isinstance(model, ScoreProfile):
        return prepare_profile(model, min_logerr=threshold, normalization=normalization, cache=cache)
    return None


def compare(query, target, sequences=None, *, background=None, metric="co", search_range=10, window_radius=10, realign_window=3, min_logerr=None, normalization=None, cache=None):
    """Compare two profiles or motif models.

    Accepts (PreparedProfile, PreparedProfile), (PreparedProfile, ScoreProfile),
    (PreparedProfile, MotifModel, sequences), (MotifModel, MotifModel, sequences),
    or (ScoreProfile, ScoreProfile).
    """
    m = parse_profile_metric(metric)

    q_prepared = isinstance(query, PreparedProfile)
    t_prepared = isinstance(target, PreparedProfile)

    if (isinstance(query, ScoreProfile) and isinstance(target, MotifModel)) or (
        isinstance(query, MotifModel) and isinstance(target, ScoreProfile)
    ):
        raise ValueError("mixed ScoreProfile/motif comparison is unsupported; prepare both inputs as profiles first.")

    if q_prepared and t_prepared:
        if query.min_logerr != target.min_logerr:
            raise ValueError("prepared profiles use different min_logerr thresholds.")
        if query.normalization != target.normalization:
            raise ValueError("prepared profiles use different normalization strategies.")
        threshold = query.min_logerr
        norm = query.normalization
    elif q_prepared or t_prepared:
        existing = query if q_prepared else target
        threshold = existing.min_logerr if min_logerr is None else np.float32(min_logerr)
        _check_threshold(threshold, existing)
        norm = existing.normalization
    else:
        threshold = np.float32(0.0 if min_logerr is None else min_logerr)
        norm = normalization
        if isinstance(query, ScoreProfile) and isinstance(target, ScoreProfile) and sequences is not None:
            raise ValueError("ScoreProfile comparison does not consume sequences.")

    pq = _prepare_side(query, sequences, background, threshold, norm, cache)
    pt = _prepare_side(target, sequences, background, threshold, norm, cache)
    if pq is None or pt is None:
        raise TypeError(f"unsupported comparison inputs: {type(query).__name__} vs {type(target).__name__}")

    config = ProfileConfig(metric=m, search_range=search_range, window_radius=window_radius, realign_window=realign_window, min_logerr=threshold)
    score, shift, orientation, n_sites, metric_str = profile_compare(
        pq.bundle, pq.anchors, pt.bundle, pt.anchors, config
    )
    return ComparisonResult(query.name, target.name, score, shift, orientation, metric_str, n_sites)


def compare_many(query, targets, sequences=None, *, background=None, metric="co", search_range=10, window_radius=10, realign_window=3, min_logerr=None, normalization=None, cache=None, on_progress=None):
    """Compare one query against targets in stable order."""
    if not isinstance(query, PreparedProfile):
        query = prepare_profile(query, sequences, background=background, min_logerr=0.0 if min_logerr is None else min_logerr, normalization=normalization, cache=cache)
    results = []
    total = len(targets)
    if on_progress is not None:
        on_progress(("compare", 0, total, ""))
    for i, target in enumerate(targets):
        results.append(
            compare(
                query,
                target,
                sequences,
                background=background,
                metric=metric,
                search_range=search_range,
                window_radius=window_radius,
                realign_window=realign_window,
                min_logerr=min_logerr,
                normalization=normalization,
                cache=cache,
            )
        )
        if on_progress is not None:
            on_progress(("compare", i + 1, total, getattr(target, "name", "")))
    return results
