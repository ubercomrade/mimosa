"""Comparison orchestration: prepare profiles and compare."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._kernels import batch_profile_compare
from .models import MotifModel
from .profiles.alignment import ProfileConfig, parse_profile_metric, profile_compare
from .profiles.prepared import (
    PreparedProfile,
    ScoreProfile,
    _prepare_profile,
    _prepare_profiles_batch,
)
from .parallel import MIN_PARALLEL_TARGETS

_TARGET_BATCH_SIZE = 32


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
            _preparation_context=preparation_context,
        )
    if isinstance(model, ScoreProfile):
        return _prepare_profile(
            model,
            min_logerr=threshold,
            normalization=normalization,
            cache=cache,
            _preparation_context=preparation_context,
        )
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

    Target preparation and bounded target batching are internal. Large prepared
    batches use one Numba target-parallel kernel; small batches stay serial.
    """
    preparation_context = None
    if cache is not None and sequences is not None:
        from .cache import _make_preparation_context

        preparation_context = _make_preparation_context(sequences, background)
    if not isinstance(query, PreparedProfile):
        query_source = query
        query = _prepare_side(
            query,
            sequences,
            background,
            0.0 if min_logerr is None else min_logerr,
            normalization,
            cache,
            preparation_context,
        )
        if query is None:
            raise TypeError(f"unsupported comparison query: {type(query_source).__name__}")

    if min_logerr is not None and np.float32(min_logerr) != query.min_logerr:
        _check_threshold(np.float32(min_logerr), query)
    threshold = query.min_logerr
    norm = query.normalization if normalization is None else normalization
    config = ProfileConfig(metric=parse_profile_metric(metric), search_range=search_range, window_radius=window_radius, realign_window=realign_window, min_logerr=threshold)

    total = len(targets)
    results = []
    if on_progress is not None:
        on_progress(("compare", 0, total, ""))
    for batch_start in range(0, total, _TARGET_BATCH_SIZE):
        target_batch = targets[batch_start : batch_start + _TARGET_BATCH_SIZE]
        prepared_targets = _prepare_profiles_batch(
            target_batch,
            sequences,
            background=background,
            min_logerr=threshold,
            normalization=norm,
            cache=cache,
            _preparation_context=preparation_context,
        )
        if any(target is None for target in prepared_targets):
            unsupported = next(
                target for target in target_batch if not isinstance(target, (PreparedProfile, MotifModel, ScoreProfile))
            )
            raise TypeError(f"unsupported comparison target: {type(unsupported).__name__}")
        if len(prepared_targets) < MIN_PARALLEL_TARGETS:
            batch_results = _compare_many_serial(query, prepared_targets, config, None)
        else:
            batch_results = _compare_many_prepared_parallel(
                query, prepared_targets, config, cache, None
            )
        results.extend(batch_results)
        if on_progress is not None:
            first_result = batch_start
            for index, result in enumerate(batch_results, first_result + 1):
                on_progress(("compare", index, total, result.target))
        del target_batch, prepared_targets, batch_results
    return results


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


def _compare_many_prepared_parallel(
    query,
    prepared_targets,
    config,
    cache,
    on_progress,
    phase_timings=None,
):
    """Compare prepared targets with one packed Numba target-parallel kernel."""
    total = len(prepared_targets)
    if on_progress is not None:
        on_progress(("compare", 0, total, ""))
    if total == 0:
        return []

    targets = list(prepared_targets)
    n_rows = len(query.bundle.forward)
    for target in targets:
        if len(target.bundle.forward) != n_rows or len(target.bundle.reverse) != n_rows:
            raise ValueError("prepared profiles must have equal row counts.")
    target_shared = np.array(
        [target.bundle.forward is target.bundle.reverse for target in targets],
        dtype=np.bool_,
    )

    def pack(strand):
        score_offsets = np.empty((total, n_rows + 1), dtype=np.int64)
        anchor_offsets = np.empty((total, n_rows + 1), dtype=np.int64)
        data_size = 0
        position_size = 0
        for index, target in enumerate(targets):
            if strand == 1 and target_shared[index]:
                score_offsets[index] = 0
                anchor_offsets[index] = 0
                continue
            scores = target.bundle.forward if strand == 0 else target.bundle.reverse
            anchors = target.anchors[0] if strand == 0 else target.anchors[1]
            data_size += scores.data.size
            position_size += anchors.positions.size
        data = np.empty(data_size, dtype=np.float32)
        positions = np.empty(position_size, dtype=np.int64)
        data_cursor = 0
        position_cursor = 0
        for index, target in enumerate(targets):
            if strand == 1 and target_shared[index]:
                continue
            scores = target.bundle.forward if strand == 0 else target.bundle.reverse
            anchors = target.anchors[0] if strand == 0 else target.anchors[1]
            score_offsets[index] = scores.offsets + data_cursor
            anchor_offsets[index] = anchors.offsets + position_cursor
            np.copyto(data[data_cursor : data_cursor + scores.data.size], scores.data)
            np.copyto(
                positions[position_cursor : position_cursor + anchors.positions.size],
                anchors.positions,
            )
            data_cursor += scores.data.size
            position_cursor += anchors.positions.size
        return data, score_offsets, positions, anchor_offsets

    if phase_timings is not None:
        import time

        pack_started = time.perf_counter()
    target_fwd = pack(0)
    target_rev = pack(1)
    if phase_timings is not None:
        phase_timings["packing"] = time.perf_counter() - pack_started
    query_shared = query.bundle.forward is query.bundle.reverse
    max_row_length = (
        int(np.max(np.diff(query.bundle.forward.offsets)))
        if query.bundle.forward.offsets.size > 1
        else 0
    )
    if config.min_logerr > 0.0:
        seen = np.zeros((total, max_row_length), dtype=np.uint32)
        candidates = np.empty((total, max_row_length), dtype=np.int64)
    else:
        seen = np.empty((0, 0), dtype=np.uint32)
        candidates = np.empty((0, 0), dtype=np.int64)
    out_scores = np.empty(total, dtype=np.float32)
    out_shifts = np.empty(total, dtype=np.int64)
    out_orientations = np.empty(total, dtype=np.int8)
    out_sites = np.empty(total, dtype=np.int64)
    score_work = np.empty(total, dtype=np.float64)
    sites_work = np.empty(total, dtype=np.int64)
    metric_kind = 1 if config.metric == "cosine" else 0
    use_dice = config.metric == "dice"

    if phase_timings is not None:
        kernel_started = time.perf_counter()
    batch_profile_compare(
        query.bundle.forward.data,
        query.bundle.forward.offsets,
        query.bundle.reverse.data,
        query.bundle.reverse.offsets,
        query.anchors[0].positions,
        query.anchors[0].offsets,
        query.anchors[1].positions,
        query.anchors[1].offsets,
        target_fwd[0],
        target_fwd[1],
        target_rev[0],
        target_rev[1],
        target_fwd[2],
        target_fwd[3],
        target_rev[2],
        target_rev[3],
        target_shared,
        query_shared,
        config.search_range,
        config.window_radius,
        config.realign_window,
        metric_kind,
        use_dice,
        config.min_logerr,
        seen,
        candidates,
        score_work,
        sites_work,
        out_scores,
        out_shifts,
        out_orientations,
        out_sites,
    )
    if phase_timings is not None:
        phase_timings["alignment_kernel"] = time.perf_counter() - kernel_started
    orientations = ("++", "+-", "-+", "--")
    results = []
    for index, target in enumerate(targets):
        results.append(
            ComparisonResult(
                query.name,
                target.name,
                out_scores[index],
                int(out_shifts[index]),
                orientations[int(out_orientations[index])],
                config.metric,
                int(out_sites[index]),
            )
        )
        if on_progress is not None:
            on_progress(("compare", index + 1, total, target.name))
    return results
