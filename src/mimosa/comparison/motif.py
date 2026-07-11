"""Motif tensor comparison strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from mimosa.cache import fingerprint_model
from mimosa.comparison.common import _cached_batch_fingerprint, _select_best_orientation, make_batch_preparation_context
from mimosa.models import GenericModel
from mimosa.sites import get_pfm
from mimosa.types import ComparatorConfig, ComparisonResult

NUCLEOTIDE_CARDINALITY = 4
AMBIGUOUS_STATE_CARDINALITY = 5
MATRIX_RANK = 2
SIMILARITY_EPS = 1e-9


def _is_power_of_four(value: int) -> bool:
    """Return True when the provided value is a positive power of four."""
    if value < 1:
        return False
    while value % NUCLEOTIDE_CARDINALITY == 0:
        value //= NUCLEOTIDE_CARDINALITY
    return value == 1


def _looks_like_alphabet_axis(size: int) -> bool:
    """Return True when one axis likely encodes nucleotide states."""
    return size in (NUCLEOTIDE_CARDINALITY, AMBIGUOUS_STATE_CARDINALITY) or _is_power_of_four(size)


def _normalize_motif_tensor(matrix: np.ndarray) -> np.ndarray:
    """Normalize one motif representation to a tensor with sequence positions last."""
    normalized = np.asarray(matrix)
    if normalized.ndim == MATRIX_RANK:
        rows, cols = normalized.shape
        if not _looks_like_alphabet_axis(rows) and _looks_like_alphabet_axis(cols):
            normalized = normalized.T
        if normalized.shape[0] == AMBIGUOUS_STATE_CARDINALITY:
            normalized = normalized[:NUCLEOTIDE_CARDINALITY, :]
        if normalized.shape[0] > NUCLEOTIDE_CARDINALITY and _is_power_of_four(normalized.shape[0]):
            order = int(round(np.log(normalized.shape[0]) / np.log(NUCLEOTIDE_CARDINALITY)))
            normalized = normalized.reshape((NUCLEOTIDE_CARDINALITY,) * order + (normalized.shape[1],))
        return normalized

    if normalized.shape[0] > AMBIGUOUS_STATE_CARDINALITY:
        normalized = np.moveaxis(normalized, 0, -1)
    clean_slice = tuple(
        slice(0, NUCLEOTIDE_CARDINALITY) if axis < normalized.ndim - 1 else slice(None)
        for axis in range(normalized.ndim)
    )
    return normalized[clean_slice]


def _reverse_complement_motif_tensor(matrix: np.ndarray) -> np.ndarray:
    """Return the reverse-complement tensor for one normalized motif matrix."""
    order = matrix.ndim - 1
    axes = tuple(range(order - 1, -1, -1)) + (order,)
    reverse = np.transpose(matrix, axes=axes)
    for axis in range(order):
        reverse = np.flip(reverse, axis=axis)
    return np.flip(reverse, axis=-1)


def _prepare_motif(matrix: np.ndarray) -> dict:
    """Build flattened forward and reverse-complement views for alignment."""
    normalized = _normalize_motif_tensor(matrix)
    reverse = _reverse_complement_motif_tensor(normalized)
    return {
        "forward": normalized.reshape(-1, normalized.shape[-1]),
        "reverse": reverse.reshape(-1, reverse.shape[-1]),
    }


def _vectorized_pcc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute column-wise Pearson correlation coefficients."""
    x1_centered = x1 - np.mean(x1, axis=0, keepdims=True)
    x2_centered = x2 - np.mean(x2, axis=0, keepdims=True)
    numerator = np.sum(x1_centered * x2_centered, axis=0)
    denominator = np.sqrt(np.sum(x1_centered**2, axis=0)) * np.sqrt(np.sum(x2_centered**2, axis=0))
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > SIMILARITY_EPS)
    return result


def _vectorized_cosine(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute column-wise cosine similarity."""
    numerator = np.sum(x1 * x2, axis=0)
    denominator = np.linalg.norm(x1, axis=0) * np.linalg.norm(x2, axis=0)
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > SIMILARITY_EPS)
    return result


def _score_motif_columns(metric: str, query_columns: np.ndarray, target_columns: np.ndarray) -> float:
    """Score one aligned column block."""
    overlap = query_columns.shape[1]
    if metric == "pcc":
        return float(np.sum(_vectorized_pcc(query_columns, target_columns)) / overlap)
    if metric == "cosine":
        return float(np.sum(_vectorized_cosine(query_columns, target_columns)) / overlap)
    if metric != "ed":
        raise ValueError("metric must be one of: 'pcc', 'ed', 'cosine'")
    distances = np.sqrt(np.sum((query_columns - target_columns) ** 2, axis=0))
    return float(-np.sum(distances) / overlap)


def _align_motif_matrices(query_matrix: np.ndarray, target_matrix: np.ndarray, metric: str) -> tuple[float, int]:
    """Align two prepared matrices and return the best score and offset."""
    query_length = query_matrix.shape[1]
    target_length = target_matrix.shape[1]
    min_overlap = min(query_length, target_length) / 2.0
    best_score = float(-np.inf)
    best_offset = 0

    for offset in range(-(target_length - 1), query_length):
        if offset < 0:
            overlap = min(query_length, target_length + offset)
            if overlap < min_overlap:
                continue
            query_slice = slice(0, overlap)
            target_slice = slice(-offset, -offset + overlap)
        else:
            overlap = min(query_length - offset, target_length)
            if overlap < min_overlap:
                continue
            query_slice = slice(offset, offset + overlap)
            target_slice = slice(0, overlap)

        score = _score_motif_columns(metric, query_matrix[:, query_slice], target_matrix[:, target_slice])
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_score, best_offset


def _score_motif_candidates(query: dict, target: dict, metric: str) -> list[dict]:
    """Score all orientation pairs for one prepared motif pair."""
    candidates = []
    for orientation, query_matrix, target_matrix in (
        ("++", query["forward"], target["forward"]),
        ("+-", query["forward"], target["reverse"]),
        ("-+", query["reverse"], target["forward"]),
        ("--", query["reverse"], target["reverse"]),
    ):
        score, offset = _align_motif_matrices(query_matrix, target_matrix, metric)
        candidates.append({"orientation": orientation, "score": score, "offset": offset})
    return candidates


def _resolve_motif_matrix(
    model: GenericModel,
    sequences,
    cfg: ComparatorConfig,
    use_pfm_mode: bool,
    runtime_cache: dict | None = None,
):
    """Resolve one motif matrix for direct or PFM-based comparison."""
    if not use_pfm_mode:
        return model.representation
    if sequences is None:
        raise ValueError("sequences are required for pfm_mode")

    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    runtime_key = ("motif_matrix", fingerprint_model(model), sequence_fp, cfg["pfm_top_fraction"])
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    matrix = get_pfm(model, sequences, top_fraction=cfg["pfm_top_fraction"])
    runtime_cache[runtime_key] = matrix
    return matrix


def _prepare_motif_model(
    model: GenericModel,
    sequences,
    cfg: ComparatorConfig,
    use_pfm_mode: bool,
    runtime_cache: dict | None = None,
):
    """Resolve one motif matrix and its prepared forward/reverse views."""
    runtime_cache = {} if runtime_cache is None else runtime_cache
    sequence_fp = _cached_batch_fingerprint(runtime_cache, sequences, "sequences")
    runtime_key = ("prepared_motif", fingerprint_model(model), use_pfm_mode, sequence_fp, cfg["pfm_top_fraction"])
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached

    matrix = _resolve_motif_matrix(model, sequences, cfg, use_pfm_mode, runtime_cache)
    prepared = _prepare_motif(matrix)
    state = (matrix, prepared)
    runtime_cache[runtime_key] = state
    return state


def _score_prepared_motif_pair(query: dict, target: dict, metric: str) -> dict:
    """Score one prepared motif pair and return the best orientation candidate."""
    return _select_best_orientation(_score_motif_candidates(query, target, metric))


def _build_motif_result(query_name: str, target_name: str, best: dict, metric: str) -> ComparisonResult:
    """Build one motif comparison result payload from the best candidate."""
    return ComparisonResult(
        query=query_name,
        target=target_name,
        score=float(best["score"]),
        offset=int(best["offset"]),
        orientation=best["orientation"],
        metric=metric,
    )


def strategy_motif(model1: GenericModel, model2: GenericModel, sequences, cfg: ComparatorConfig) -> ComparisonResult:
    """Matrix-based comparison strategy (PCC/ED/Cosine)."""
    runtime_cache: dict[Any, Any] = {}
    use_pfm_mode = cfg["pfm_mode"] or (model1.type_key != model2.type_key)
    _query_matrix, prepared1 = _prepare_motif_model(model1, sequences, cfg, use_pfm_mode, runtime_cache)
    _matrix2, prepared2 = _prepare_motif_model(model2, sequences, cfg, use_pfm_mode, runtime_cache)
    best = _score_prepared_motif_pair(prepared1, prepared2, cfg["metric"])
    return _build_motif_result(model1.name, model2.name, best, cfg["metric"])


def _compare_motif_one_to_many(  # noqa: PLR0913
    query_model: GenericModel,
    target_models,
    sequences,
    cfg: ComparatorConfig,
    *,
    progress: bool | None = False,
    progress_desc: str | None = None,
    progress_leave: bool = True,
) -> list[ComparisonResult]:
    """Compare one motif query against many targets while reusing prepared query state."""
    from mimosa.comparison.runner import _run_target_comparisons

    target_list = list(target_models)
    if not target_list:
        return []

    query_cache: dict[Any, Any] = make_batch_preparation_context(sequences, None)
    use_pfm_modes = {
        bool(cfg["pfm_mode"] or (query_model.type_key != target_model.type_key)) for target_model in target_list
    }
    prepared_query_by_mode: dict[bool, Any] = {}
    for use_pfm_mode in use_pfm_modes:
        _query_matrix, prepared_query = _prepare_motif_model(query_model, sequences, cfg, use_pfm_mode, query_cache)
        prepared_query_by_mode[use_pfm_mode] = prepared_query

    def _score_target(target_model: GenericModel) -> ComparisonResult:
        target_cache: dict[Any, Any] = dict(query_cache)
        try:
            use_pfm_mode = bool(cfg["pfm_mode"] or (query_model.type_key != target_model.type_key))
            prepared_query = prepared_query_by_mode[use_pfm_mode]
            _target_matrix, prepared_target = _prepare_motif_model(
                target_model,
                sequences,
                cfg,
                use_pfm_mode,
                target_cache,
            )
            best = _score_prepared_motif_pair(prepared_query, prepared_target, cfg["metric"])
            return _build_motif_result(query_model.name, target_model.name, best, cfg["metric"])
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
