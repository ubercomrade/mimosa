"""Comparison orchestration: prepare profiles and compare."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from .models import BaMM, Dimont, MotifModel, PWM, SiteGA, Slim
from .profiles.alignment import METRIC_NAME, ProfileConfig, profile_compare
from .profiles.anchors import DEFAULT_MIN_LOGERR, parse_min_logerr
from .profiles.prepared import (
    PreparedProfile,
    ScoreProfile,
    _prepare_profile,
)


@dataclass(frozen=True)
class ComparisonResult:
    query: str
    target: str
    score: np.float32
    offset: int
    orientation: str
    metric: str = METRIC_NAME
    n_sites: int = 0
    # Comparison parameters that bound the validity of null-based p-values;
    # they are not part of the serialized result or of equality.
    search_range: int | None = field(default=None, compare=False, repr=False)
    min_logerr: np.float32 | None = field(default=None, compare=False, repr=False)

    def to_dict(self):
        d = {
            "query": self.query,
            "target": self.target,
            "score": float(self.score),
            "offset": self.offset,
            "orientation": self.orientation,
            "metric": self.metric,
            "n_sites": int(self.n_sites),
        }
        return d


def _check_threshold(threshold, prepared):
    if threshold != prepared.min_logerr:
        raise ValueError("min_logerr differs from the prepared query threshold.")


def _check_input_kinds(query, target):
    if (isinstance(query, ScoreProfile) and isinstance(target, MotifModel)) or (
        isinstance(query, MotifModel) and isinstance(target, ScoreProfile)
    ):
        raise ValueError("mixed ScoreProfile/motif comparison is unsupported; prepare both inputs as profiles first.")


def _resolve_threshold_and_normalization(prepared, min_logerr, normalization):
    """Take threshold and normalization from prepared inputs, else defaults."""
    if not prepared:
        threshold = parse_min_logerr(DEFAULT_MIN_LOGERR if min_logerr is None else min_logerr)
        return threshold, normalization
    existing = prepared[0]
    threshold = existing.min_logerr if min_logerr is None else parse_min_logerr(min_logerr)
    _check_threshold(threshold, existing)
    if normalization is not None and normalization != existing.normalization:
        raise ValueError("prepared query and requested normalization differ.")
    return threshold, existing.normalization


def _check_prepared_compatibility(query, target):
    if query.min_logerr != target.min_logerr:
        raise ValueError("prepared profiles use different min_logerr thresholds.")
    if query.normalization != target.normalization:
        raise ValueError("prepared profiles use different normalization strategies.")
    _check_circular(query)
    _check_circular(target)


def _check_circular(profile):
    if isinstance(profile, PreparedProfile) and not profile.circular:
        raise ValueError(
            "comparison requires circular prepared profiles; "
            "use prepare_profile(..., circular=True)."
        )


def _warn_if_few_positions(prepared):
    """Warn when too few positions per strand are expected above the threshold.

    Compared profiles share row layout and threshold, so checking the query
    covers every target.
    """
    positions = prepared.bundle.forward.data.size
    expected = positions * 10.0 ** -float(prepared.min_logerr)
    if expected < MIN_EXPECTED_ANCHORS:
        warnings.warn(
            f"profile '{prepared.name}': only about {expected:.3g} of {positions} "
            f"positions per strand are expected at "
            f"min_logerr={float(prepared.min_logerr):g}; the {METRIC_NAME} "
            "score is noisy. Use more sequence positions or a lower min_logerr.",
            stacklevel=3,
        )


def _compare_prepared(query, target, config):
    score, shift, orientation, n_sites = profile_compare(
        query.bundle,
        query.anchors,
        target.bundle,
        target.anchors,
        config,
    )
    return ComparisonResult(
        query.name,
        target.name,
        score,
        shift,
        orientation,
        METRIC_NAME,
        n_sites,
        search_range=config.search_range,
        min_logerr=config.min_logerr,
    )


_BUILTIN_MODELS = (PWM, BaMM, Dimont, SiteGA, Slim)
MIN_EXPECTED_ANCHORS = 50
_worker_cache_path: str | None = None
_worker_cache_obj = None


@contextmanager
def _numba_thread_budget(threads):
    from numba import get_num_threads, set_num_threads

    previous = get_num_threads()
    set_num_threads(threads)
    try:
        yield
    finally:
        set_num_threads(previous)


def _worker_cache(directory):
    global _worker_cache_path, _worker_cache_obj
    if directory is None:
        return None
    if _worker_cache_path == directory:
        return _worker_cache_obj
    from .cache import Cache

    _worker_cache_path = directory
    _worker_cache_obj = Cache(directory)
    return _worker_cache_obj


def _prepare_and_compare_with_threads(
    query,
    target_source,
    sequences,
    background,
    threshold,
    normalization,
    config,
    cache_directory,
    preparation_context,
    threads,
):
    with _numba_thread_budget(threads):
        target = _prepare_side(
            target_source,
            sequences,
            background=background,
            threshold=threshold,
            normalization=normalization,
            cache=_worker_cache(cache_directory),
            preparation_context=preparation_context,
        )
        if target is None:
            raise TypeError(f"unsupported comparison target: {type(target_source).__name__}")
        _check_prepared_compatibility(query, target)
        return _compare_prepared(query, target, config)


def _prepare_side(
    model,
    sequences,
    background,
    threshold,
    normalization,
    cache,
    preparation_context=None,
):
    if isinstance(model, PreparedProfile):
        return model
    if isinstance(model, MotifModel):
        if sequences is None:
            raise ValueError("motif comparison requires comparison sequences.")
        return _prepare_profile(
            model,
            sequences,
            background=background,
            min_logerr=threshold,
            normalization=normalization,
            cache=cache,
            circular=True,
            _preparation_context=preparation_context,
        )
    if isinstance(model, ScoreProfile):
        return _prepare_profile(
            model,
            min_logerr=threshold,
            normalization=normalization,
            cache=cache,
            circular=True,
            _preparation_context=preparation_context,
        )
    return None


def compare(query, target, sequences=None, *, background=None, search_range=5, min_logerr=None, normalization=None, cache=None):
    """Compare two profiles or motif models by weighted Jaccard similarity.

    Accepts (PreparedProfile, PreparedProfile), (PreparedProfile, ScoreProfile),
    (PreparedProfile, MotifModel, sequences), (MotifModel, MotifModel, sequences),
    or (ScoreProfile, ScoreProfile). Sequences are treated as circular, so
    prepared inputs must come from ``prepare_profile(..., circular=True)``.
    ``min_logerr=None`` takes the threshold of a prepared input, else 4.0.
    """
    q_prepared = isinstance(query, PreparedProfile)
    t_prepared = isinstance(target, PreparedProfile)
    _check_circular(query)
    _check_circular(target)

    _check_input_kinds(query, target)

    if q_prepared and t_prepared:
        _check_prepared_compatibility(query, target)
    threshold, norm = _resolve_threshold_and_normalization(
        [x for x in (query, target) if isinstance(x, PreparedProfile)],
        min_logerr,
        normalization,
    )
    if not (q_prepared or t_prepared):
        if isinstance(query, ScoreProfile) and isinstance(target, ScoreProfile) and sequences is not None:
            raise ValueError("ScoreProfile comparison does not consume sequences.")

    config = ProfileConfig(search_range=search_range, min_logerr=threshold)
    preparation_context = None
    if cache is not None and sequences is not None:
        from .cache import _make_preparation_context

        preparation_context = _make_preparation_context(sequences, background)
    pq = _prepare_side(
        query,
        sequences,
        background,
        threshold,
        norm,
        cache,
        preparation_context,
    )
    pt = _prepare_side(
        target,
        sequences,
        background,
        threshold,
        norm,
        cache,
        preparation_context,
    )
    if pq is None or pt is None:
        raise TypeError(f"unsupported comparison inputs: {type(query).__name__} vs {type(target).__name__}")
    _check_prepared_compatibility(pq, pt)
    _warn_if_few_positions(pq)
    return _compare_prepared(pq, pt, config)


def compare_many(
    query,
    targets,
    sequences=None,
    *,
    background=None,
    search_range=5,
    min_logerr=None,
    normalization=None,
    cache=None,
    total_threads=1,
    inner_threads=1,
):
    """Compare one query against targets in stable order.

    ``total_threads`` is divided between joblib target workers and Numba
    threads. Target preparation and alignment run in the same worker.
    """
    if isinstance(total_threads, bool) or not isinstance(total_threads, (int, np.integer)):
        raise TypeError("total_threads must be a positive integer.")
    if total_threads < 1:
        raise ValueError("total_threads must be a positive integer.")
    total_threads = int(total_threads)
    if isinstance(inner_threads, bool) or not isinstance(inner_threads, (int, np.integer)):
        raise TypeError("inner_threads must be an integer from 1 through 4.")
    if not 1 <= inner_threads <= 4:
        raise ValueError("inner_threads must be an integer from 1 through 4.")
    inner_threads = int(inner_threads)
    if total_threads % inner_threads:
        raise ValueError("total_threads must be divisible by inner_threads.")
    target_sources = list(targets)
    _check_circular(query)
    for target in target_sources:
        _check_circular(target)
    joblib_workers = min(total_threads // inner_threads, len(target_sources)) if target_sources else 1
    if joblib_workers > 1:
        if isinstance(query, MotifModel) and not isinstance(query, _BUILTIN_MODELS):
            raise TypeError(
                "custom models must be prepared before parallel compare_many or compared serially."
            )
        if any(
            isinstance(target, MotifModel) and not isinstance(target, _BUILTIN_MODELS)
            for target in target_sources
        ):
            raise TypeError(
                "custom models must be prepared before parallel compare_many or compared serially."
            )
    preparation_context = None
    if cache is not None and sequences is not None:
        from .cache import _make_preparation_context

        preparation_context = _make_preparation_context(sequences, background)
    for target in target_sources:
        _check_input_kinds(query, target)
    prepared_inputs = [
        x for x in (query, *target_sources) if isinstance(x, PreparedProfile)
    ]
    threshold, normalization = _resolve_threshold_and_normalization(
        prepared_inputs, min_logerr, normalization
    )
    config = ProfileConfig(search_range=search_range, min_logerr=threshold)
    if not isinstance(query, PreparedProfile):
        query_source = query
        with _numba_thread_budget(inner_threads):
            query = _prepare_side(
                query,
                sequences,
                background,
                threshold,
                normalization,
                cache,
                preparation_context,
            )
        if query is None:
            raise TypeError(f"unsupported comparison query: {type(query_source).__name__}")
    if normalization is not None and normalization != query.normalization:
        raise ValueError("prepared query and requested normalization differ.")
    norm = query.normalization
    _warn_if_few_positions(query)

    if joblib_workers > 1:
        return Parallel(n_jobs=joblib_workers, backend="loky")(
            delayed(_prepare_and_compare_with_threads)(
                query,
                target_source,
                sequences,
                background,
                threshold,
                norm,
                config,
                None if cache is None else cache.directory,
                preparation_context,
                int(inner_threads),
            )
            for target_source in target_sources
        )

    # Serial mode keeps preparation and comparison in one target-at-a-time pipeline.
    results = []
    with _numba_thread_budget(inner_threads):
        for target_source in target_sources:
            target = _prepare_side(
                target_source,
                sequences,
                background=background,
                threshold=threshold,
                normalization=norm,
                cache=cache,
                preparation_context=preparation_context,
            )
            if target is None:
                raise TypeError(
                    f"unsupported comparison target: {type(target_source).__name__}"
                )
            _check_prepared_compatibility(query, target)
            results.append(_compare_prepared(query, target, config))
    return results
