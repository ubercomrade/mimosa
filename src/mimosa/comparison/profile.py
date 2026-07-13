"""Profile-based comparison strategy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numba import get_num_threads, njit

from mimosa.batches import flatten_profile_bundle
from mimosa.cache import ProfileCacheSpec, fingerprint_model, load_profile_cache, store_profile_cache
from mimosa.comparison.common import _cached_batch_fingerprint, _select_best_orientation
from mimosa.comparison.config import SUPPORTED_PROFILE_METRICS
from mimosa.functions import (
    apply_score_log_tail_table_to_profile_bundle,
    build_score_log_tail_table,
    calc_co,
    calc_dice,
    prepare_profile_bundle,
    rowwise_co,
    rowwise_cosine,
    rowwise_dice,
    scores_to_empirical_log_tail_bundle,
)
from mimosa.functions.alignment import (
    build_anchor_csr,
    make_alignment_workspace,
    score_shift,
    should_use_parallel,
)
from mimosa.models import GenericModel
from mimosa.scanning import scan_model_strands
from mimosa.types import ComparatorConfig, ComparisonResult

logger = logging.getLogger(__name__)

PROFILE_ORIENTATION_PAIRS = (
    ("++", 0, 0),
    ("--", 1, 1),
    ("+-", 0, 1),
    ("-+", 1, 0),
)


def _get_profile_background_sequences(sequences, cfg: ComparatorConfig):
    """Return the sequence collection used to fit profile normalization."""
    return cfg.get("background") if cfg.get("background") is not None else sequences


def _resolve_raw_profile_bundle(model: GenericModel, sequences, runtime_cache: dict | None = None):
    """Resolve one raw strand-aware profile bundle before normalization."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    runtime_key = ("raw_profile_bundle", fingerprint_model(model), sequence_fp)
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    if model.type_key != "scores" and sequences is None:
        raise ValueError("Profile strategy requires sequences when comparing motif models.")

    profile_bundle = scan_model_strands(model, sequences)
    runtime_cache[runtime_key] = profile_bundle
    return profile_bundle


def _fit_profile_normalizer(
    model: GenericModel, background_sequences, cfg: ComparatorConfig, runtime_cache: dict | None = None
):
    """Fit normalization parameters from the calibration score sample."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    background_fp = _cached_batch_fingerprint(runtime_cache, background_sequences, "background")
    runtime_key = ("profile_normalizer", fingerprint_model(model), cfg["profile_normalization"], background_fp)
    if runtime_key in runtime_cache:
        return runtime_cache[runtime_key]

    background_bundle = _resolve_raw_profile_bundle(model, background_sequences, runtime_cache)
    calibration_sample = flatten_profile_bundle(background_bundle)
    if cfg["profile_normalization"] != "empirical_log_tail":
        raise ValueError(f"Unsupported profile normalization: {cfg['profile_normalization']}")

    normalizer = build_score_log_tail_table(calibration_sample)
    runtime_cache[runtime_key] = normalizer
    return normalizer


def _apply_profile_normalizer(profile_bundle, normalizer, profile_normalization: str):
    """Apply fitted normalization parameters to one raw profile bundle."""
    if profile_normalization != "empirical_log_tail":
        raise ValueError(f"Unsupported profile normalization: {profile_normalization}")
    return apply_score_log_tail_table_to_profile_bundle(profile_bundle, normalizer)


def _build_profile_cache_spec(
    model: GenericModel, sequences, background_sequences, cfg: ComparatorConfig, profile_kind: str
) -> ProfileCacheSpec:
    """Build one cache descriptor for a normalized profile bundle."""
    return {
        "model": model,
        "sequences": sequences,
        "background": background_sequences,
        "profile_kind": profile_kind,
        "cache_dir": cfg["cache_dir"],
    }


def _resolve_profile_bundle(
    model: GenericModel, sequences, background_sequences, cfg: ComparatorConfig, runtime_cache: dict | None = None
):
    """Resolve one model to the normalized strand-aware profile bundle used in profile comparisons."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    profile_kind = cfg["profile_normalization"]
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    background_fp = _cached_batch_fingerprint(runtime_cache, background_sequences, "background")
    runtime_key = (fingerprint_model(model), profile_kind, sequence_fp, background_fp)

    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    cache_spec = None
    if cfg["cache_mode"] == "on":
        cache_spec = _build_profile_cache_spec(model, sequences, background_sequences, cfg, profile_kind)
        cached = load_profile_cache(cache_spec)
        if cached is not None:
            runtime_cache[runtime_key] = cached
            logger.debug("Profile cache hit for model '%s'.", model.name)
            return cached

    raw_bundle = _resolve_raw_profile_bundle(model, sequences, runtime_cache)
    if background_sequences is sequences and profile_kind == "empirical_log_tail":
        profile_bundle = scores_to_empirical_log_tail_bundle(raw_bundle)
        runtime_cache[runtime_key] = profile_bundle
        if cache_spec is not None:
            store_profile_cache(cache_spec, profile_bundle)
            logger.debug("Stored profile cache for model '%s'.", model.name)
        return profile_bundle

    normalizer = _fit_profile_normalizer(model, background_sequences, cfg, runtime_cache)
    profile_bundle = _apply_profile_normalizer(raw_bundle, normalizer, profile_kind)
    runtime_cache[runtime_key] = profile_bundle

    if cache_spec is not None:
        store_profile_cache(cache_spec, profile_bundle)
        logger.debug("Stored profile cache for model '%s'.", model.name)

    return profile_bundle


def _prepare_profile_model(
    model: GenericModel,
    sequences,
    background_sequences,
    cfg: ComparatorConfig,
    runtime_cache: dict | None = None,
):
    """Resolve one normalized profile bundle in a contiguous scoring layout."""
    bundle = _resolve_profile_bundle(model, sequences, background_sequences, cfg, runtime_cache)
    return prepare_profile_bundle(bundle)


def _empty_positions() -> tuple[np.ndarray, np.ndarray]:
    """Return one empty anchor payload."""
    empty = np.empty(0, dtype=np.int32)
    return empty, empty


@njit(cache=False, nogil=False)
def _collect_best_anchor_positions_numba(scores: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collect one best anchor per row."""
    n_rows = scores.shape[0]
    rows = np.empty(n_rows, dtype=np.int32)
    positions = np.empty(n_rows, dtype=np.int32)
    out_index = 0

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        if length <= 0:
            continue

        best_position = 0
        best_score = scores[row_index, 0]
        for pos in range(1, length):
            score = scores[row_index, pos]
            if score > best_score:
                best_score = score
                best_position = pos

        rows[out_index] = row_index
        positions[out_index] = best_position
        out_index += 1

    return rows[:out_index], positions[:out_index]


@njit(cache=False, nogil=False)
def _count_threshold_anchor_positions_numba(scores: np.ndarray, lengths: np.ndarray, score_threshold: float) -> int:
    """Count threshold-selected anchors."""
    total = 0
    for row_index in range(scores.shape[0]):
        length = int(lengths[row_index])
        for pos in range(length):
            if scores[row_index, pos] >= score_threshold:
                total += 1
    return total


@njit(cache=False, nogil=False)
def _collect_threshold_anchor_positions_numba(
    scores: np.ndarray,
    lengths: np.ndarray,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all anchors at or above the configured threshold."""
    total = _count_threshold_anchor_positions_numba(scores, lengths, score_threshold)
    rows = np.empty(total, dtype=np.int32)
    positions = np.empty(total, dtype=np.int32)
    out_index = 0

    for row_index in range(scores.shape[0]):
        length = int(lengths[row_index])
        for pos in range(length):
            if scores[row_index, pos] >= score_threshold:
                rows[out_index] = row_index
                positions[out_index] = pos
                out_index += 1

    return rows, positions


def _collect_anchor_sites(
    scores: np.ndarray,
    lengths: np.ndarray,
    score_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect site anchors in best or threshold mode for one strand-specific score matrix."""
    scores_array = np.ascontiguousarray(scores, dtype=np.float32)
    lengths_array = np.ascontiguousarray(lengths, dtype=np.int32)
    if scores_array.shape[0] == 0:
        return _empty_positions()
    if score_threshold is None:
        return _collect_best_anchor_positions_numba(scores_array, lengths_array)
    return _collect_threshold_anchor_positions_numba(scores_array, lengths_array, float(score_threshold))


@njit(cache=False, nogil=False)
def _collect_model2_window_candidates_numba(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    lengths2: np.ndarray,
    anchor_rows: np.ndarray,
    anchor_pos2: np.ndarray,
    shift: int,
    realign_window: int,
    min_offset: int,
    max_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect valid windows centered on model2 anchors and realigned on model1."""
    size = anchor_rows.shape[0]
    rows = np.empty(size, dtype=np.int32)
    pos1 = np.empty(size, dtype=np.int32)
    pos2 = np.empty(size, dtype=np.int32)
    out_index = 0

    for candidate_index in range(size):
        row = int(anchor_rows[candidate_index])
        row_length1 = int(lengths1[row])
        row_length2 = int(lengths2[row])
        if row_length1 <= 0 or row_length2 <= 0:
            continue

        expected_pos1 = int(anchor_pos2[candidate_index]) - shift
        left = max(0, expected_pos1 - realign_window)
        right = min(row_length1 - 1, expected_pos1 + realign_window)
        if left > right:
            continue

        best_pos1 = left
        best_score = scores1[row, left]
        for pos in range(left + 1, right + 1):
            score = scores1[row, pos]
            if score > best_score:
                best_score = score
                best_pos1 = pos

        aligned_pos2 = best_pos1 + shift
        if (
            best_pos1 + min_offset < 0
            or best_pos1 + max_offset >= row_length1
            or aligned_pos2 + min_offset < 0
            or aligned_pos2 + max_offset >= row_length2
        ):
            continue

        rows[out_index] = row
        pos1[out_index] = best_pos1
        pos2[out_index] = aligned_pos2
        out_index += 1

    return rows[:out_index], pos1[:out_index], pos2[:out_index]


def _collect_model2_window_candidates(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    lengths2: np.ndarray,
    anchor_rows: np.ndarray,
    anchor_pos2: np.ndarray,
    shift: int,
    min_offset: int,
    max_offset: int,
    realign_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect valid windows centered on model2 anchors and realigned on model1."""
    if anchor_rows.size == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty

    return _collect_model2_window_candidates_numba(
        np.ascontiguousarray(scores1, dtype=np.float32),
        np.ascontiguousarray(lengths1, dtype=np.int32),
        np.ascontiguousarray(lengths2, dtype=np.int32),
        np.ascontiguousarray(anchor_rows, dtype=np.int32),
        np.ascontiguousarray(anchor_pos2, dtype=np.int32),
        int(shift),
        int(realign_window),
        int(min_offset),
        int(max_offset),
    )


def _score_window_collection(metric: str, windows1: np.ndarray, windows2: np.ndarray) -> float:
    """Score explicit windows for metric-level compatibility tests."""
    if windows1.shape != windows2.shape:
        raise ValueError("Window collections must have identical shapes.")
    if windows1.size == 0:
        return 0.0
    if metric == "co":
        return calc_co(windows1, windows2)
    if metric == "co_rowwise":
        return _mean_finite_row_scores(rowwise_co(windows1, windows2))
    if metric == "dice":
        return calc_dice(windows1, windows2)
    if metric == "dice_rowwise":
        return _mean_finite_row_scores(rowwise_dice(windows1, windows2))
    if metric != "cosine":
        options = ", ".join(repr(metric_name) for metric_name in SUPPORTED_PROFILE_METRICS)
        raise ValueError(f"metric must be one of: {options}")
    return _mean_finite_row_scores(rowwise_cosine(windows1, windows2))


def _mean_finite_row_scores(values: np.ndarray) -> float:
    """Average finite row-wise window scores and treat all-masked inputs as zero."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _compute_shifted_window_alignment(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    scores2: np.ndarray,
    lengths2: np.ndarray,
    shift: int,
    offsets: np.ndarray,
    min_offset: int,
    max_offset: int,
    query_anchors: tuple[np.ndarray, np.ndarray],
    target_anchors: tuple[np.ndarray, np.ndarray],
    realign_window: int,
    metric: str,
) -> dict[str, int | float]:
    """Evaluate one shift through the fused kernel (compatibility adapter)."""
    del offsets, min_offset
    n_rows = int(scores1.shape[0])
    query_csr = build_anchor_csr(query_anchors[0], query_anchors[1], n_rows)
    target_csr = build_anchor_csr(target_anchors[0], target_anchors[1], n_rows)
    workspace = make_alignment_workspace(n_rows, int(scores1.shape[1]))
    score, n_sites = score_shift(
        np.ascontiguousarray(scores1, dtype=np.float32),
        np.ascontiguousarray(lengths1, dtype=np.int32),
        np.ascontiguousarray(scores2, dtype=np.float32),
        np.ascontiguousarray(lengths2, dtype=np.int32),
        query_csr,
        target_csr,
        int(shift),
        int(max_offset),
        int(realign_window),
        metric,
        workspace,
        1,
        False,
    )
    return {
        "score": score,
        "shift": int(shift),
        "n_sites": n_sites,
    }


def _score_profile_orientation_pair(
    query_bundle: dict,
    target_bundle: dict,
    query_strand: int,
    target_strand: int,
    query_anchors: tuple[np.ndarray, np.ndarray],
    target_anchors: tuple[np.ndarray, np.ndarray],
    cfg: ComparatorConfig,
) -> dict:
    """Score one profile orientation across all tested shifts."""
    query_scores = query_bundle["values"][query_strand]
    target_scores = target_bundle["values"][target_strand]
    query_lengths = query_bundle["lengths"]
    target_lengths = target_bundle["lengths"]

    if query_scores.shape[0] != target_scores.shape[0]:
        raise ValueError("Profile bundles must have the same number of rows.")

    search_range = int(cfg["search_range"])
    window_radius = int(cfg["window_radius"])
    workspace = make_alignment_workspace(int(query_scores.shape[0]), int(query_scores.shape[1]))
    use_parallel = should_use_parallel(
        int(query_scores.shape[0]), int(query_scores.shape[1]), search_range, get_num_threads()
    )
    best = {"score": 0.0, "shift": 0, "n_sites": 0}
    for generation, shift in enumerate(range(-search_range, search_range + 1), start=1):
        score, n_sites = score_shift(
            query_scores,
            query_lengths,
            target_scores,
            target_lengths,
            query_anchors,
            target_anchors,
            shift,
            window_radius,
            int(cfg["realign_window"]),
            str(cfg["metric"]),
            workspace,
            generation,
            use_parallel,
        )
        candidate = {"score": score, "shift": shift, "n_sites": n_sites}
        if float(candidate["score"]) > float(best["score"]) or (
            float(candidate["score"]) == float(best["score"])
            and (
                int(candidate["n_sites"]) > int(best["n_sites"])
                or (
                    int(candidate["n_sites"]) == int(best["n_sites"])
                    and abs(int(candidate["shift"])) < abs(int(best["shift"]))
                )
            )
        ):
            best = candidate

    return {
        "score": float(best["score"]),
        "shift": int(best["shift"]),
        "n_sites": int(best["n_sites"]),
        "target_strand": int(target_strand),
    }


def _score_profile_candidates(query_bundle: dict, target_bundle: dict, pair_specs, cfg: ComparatorConfig) -> list[dict]:
    """Score all requested orientation pairs with the window-based profile algorithm."""
    min_logfpr = cfg["min_logfpr"]
    score_threshold = None if min_logfpr is None or float(min_logfpr) <= 0.0 else float(min_logfpr)
    n_rows = int(query_bundle["values"].shape[1])
    query_strands = {int(query_strand) for _, query_strand, _ in pair_specs}
    target_strands = {int(target_strand) for _, _, target_strand in pair_specs}
    query_anchor_cache = {
        strand_index: build_anchor_csr(
            *_collect_anchor_sites(query_bundle["values"][strand_index], query_bundle["lengths"], score_threshold),
            n_rows,
        )
        for strand_index in query_strands
    }
    target_anchor_cache = {
        strand_index: build_anchor_csr(
            *_collect_anchor_sites(target_bundle["values"][strand_index], target_bundle["lengths"], score_threshold),
            n_rows,
        )
        for strand_index in target_strands
    }
    candidates = []
    for orientation, query_strand, target_strand in pair_specs:
        best = _score_profile_orientation_pair(
            query_bundle,
            target_bundle,
            int(query_strand),
            int(target_strand),
            query_anchor_cache[int(query_strand)],
            target_anchor_cache[int(target_strand)],
            cfg,
        )
        best["orientation"] = orientation
        candidates.append(best)
    return candidates


def _build_profile_result(query_name: str, target_name: str, best: dict, metric: str) -> ComparisonResult:
    """Build one profile comparison result payload from the best candidate."""
    return ComparisonResult(
        query=query_name,
        target=target_name,
        score=float(best["score"]),
        offset=int(best["shift"]),
        orientation=best["orientation"],
        metric=metric,
        n_sites=int(best["n_sites"]),
    )


def strategy_profile(model1: GenericModel, model2: GenericModel, sequences, cfg: ComparatorConfig) -> ComparisonResult:
    """Window-based profile comparison strategy (CO/rowwise-CO/Dice/Cosine similarity)."""
    runtime_cache: dict[Any, Any] = {}
    background_sequences = _get_profile_background_sequences(sequences, cfg)
    bundle1 = _prepare_profile_model(model1, sequences, background_sequences, cfg, runtime_cache)
    bundle2 = _prepare_profile_model(model2, sequences, background_sequences, cfg, runtime_cache)
    best = _select_best_orientation(_score_profile_candidates(bundle1, bundle2, PROFILE_ORIENTATION_PAIRS, cfg))
    return _build_profile_result(model1.name, model2.name, best, cfg["metric"])


def _compare_profile_one_to_many(
    query_model: GenericModel,
    target_models,
    sequences,
    cfg: ComparatorConfig,
    *,
    progress: bool | None = False,
    progress_desc: str | None = None,
    progress_leave: bool = True,
) -> list[ComparisonResult]:
    """Compare one profile query against many targets while reusing normalized query profiles."""
    from mimosa.comparison.runner import _run_target_comparisons

    target_list = list(target_models)
    if not target_list:
        return []

    query_cache: dict[Any, Any] = {}
    background_sequences = _get_profile_background_sequences(sequences, cfg)
    query_bundle = _prepare_profile_model(
        query_model,
        sequences,
        background_sequences,
        cfg,
        query_cache,
    )

    def _score_target(target_model: GenericModel) -> ComparisonResult:
        target_cache: dict[Any, Any] = {}
        try:
            target_bundle = _prepare_profile_model(
                target_model,
                sequences,
                background_sequences,
                cfg,
                target_cache,
            )
            best = _select_best_orientation(
                _score_profile_candidates(query_bundle, target_bundle, PROFILE_ORIENTATION_PAIRS, cfg)
            )
            return _build_profile_result(query_model.name, target_model.name, best, cfg["metric"])
        finally:
            target_cache.clear()

    return _run_target_comparisons(
        target_list,
        cfg["n_jobs"],
        _score_target,
        progress=progress,
        progress_desc=progress_desc,
        progress_leave=progress_leave,
    )
