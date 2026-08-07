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
    """Compare one query against targets in stable order.

    Targets below MIN_PARALLEL_TARGETS run in a serial loop; larger batches
    spread across processes (one serial numba thread per worker).
    """
    if not isinstance(query, PreparedProfile):
        query = prepare_profile(query, sequences, background=background, min_logerr=0.0 if min_logerr is None else min_logerr, normalization=normalization, cache=cache)

    if min_logerr is not None and np.float32(min_logerr) != query.min_logerr:
        _check_threshold(np.float32(min_logerr), query)
    threshold = query.min_logerr
    norm = query.normalization if normalization is None else normalization
    config = ProfileConfig(metric=parse_profile_metric(metric), search_range=search_range, window_radius=window_radius, realign_window=realign_window, min_logerr=threshold)

    from .parallel import use_process_pool

    if not use_process_pool(len(targets)):
        prepared_targets = [_prepare_side(t, sequences, background, threshold, norm, cache) for t in targets]
        return _compare_many_serial(query, prepared_targets, config, on_progress)
    return _compare_many_parallel(query, targets, config, sequences, background, norm, cache, on_progress)


def _compare_many_serial(query, prepared_targets, config, on_progress):
    total = len(prepared_targets)
    results = []
    if on_progress is not None:
        on_progress(("compare", 0, total, ""))
    for i, target in enumerate(prepared_targets):
        score, shift, orientation, n_sites, metric_str = profile_compare(
            query.bundle, query.anchors, target.bundle, target.anchors, config
        )
        results.append(ComparisonResult(query.name, target.name, score, shift, orientation, metric_str, n_sites))
        if on_progress is not None:
            on_progress(("compare", i + 1, total, target.name))
    return results


def _compare_many_parallel(query, targets, config, sequences, background, normalization, cache, on_progress):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ._worker import _init_worker, _prepare_and_compare

    cache_dir = cache.directory if cache is not None else None
    total = len(targets)
    results = [None] * total
    if on_progress is not None:
        on_progress(("compare", 0, total, ""))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=mp.cpu_count(),
        initializer=_init_worker,
        initargs=(query, config, sequences, background, cache_dir, normalization),
        mp_context=ctx,
    ) as ex:
        futures = {ex.submit(_prepare_and_compare, t): i for i, t in enumerate(targets)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            name, score, shift, orientation, n_sites, metric_str = fut.result()
            results[idx] = ComparisonResult(query.name, name, score, shift, orientation, metric_str, n_sites)
            done += 1
            if on_progress is not None:
                on_progress(("compare", done, total, name))
    return results
