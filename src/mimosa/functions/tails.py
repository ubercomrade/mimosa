"""Empirical score-tail normalization helpers."""

from __future__ import annotations

import numpy as np
from numba import njit

from mimosa.batches import (
    SCORE_PADDING,
    batch_with_values,
    flatten_profile_bundle,
    flatten_valid,
    pack_batch,
    pack_profile_bundle,
)


def build_score_log_tail_table(scores: np.ndarray) -> np.ndarray:
    """Build a score-to-log-tail lookup table from one score sample."""
    flat = np.asarray(scores, dtype=np.float32).ravel()
    if flat.size == 0:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    scores_sorted = np.sort(flat)[::-1]
    unique_scores, counts = np.unique(scores_sorted, return_counts=True)
    unique_scores = unique_scores[::-1]
    counts = counts[::-1]

    cum_counts = np.cumsum(counts)
    tail_probabilities = cum_counts / flat.size
    log_tail = -np.log10(tail_probabilities)
    return np.column_stack([unique_scores, log_tail]).astype(np.float32, copy=False)


@njit(cache=False, nogil=False)
def _lower_bound_desc(values, target):
    """Find the first descending-table index whose score is not greater than target."""
    size = values.shape[0]
    if size <= 1 or target >= values[0]:
        return 0
    if target <= values[size - 1]:
        return size - 1

    lo = 0
    hi = size
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if values[mid] > target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=False, nogil=False)
def _apply_score_log_tail_table_numba(values, mask, scores_col, log_tail_col, padding_value: float):
    """Map one dense masked score matrix to empirical log-tail values."""
    rows, cols = values.shape
    mapped = np.empty((rows, cols), dtype=np.float32)

    for row_index in range(rows):
        for col_index in range(cols):
            if mask[row_index, col_index]:
                idx = _lower_bound_desc(scores_col, values[row_index, col_index])
                mapped[row_index, col_index] = log_tail_col[idx]
            else:
                mapped[row_index, col_index] = padding_value

    return mapped


def apply_score_log_tail_table(score_batch, table: np.ndarray):
    """Map one score batch to empirical log-tail values using a lookup table."""
    table_arr = np.asarray(table, dtype=np.float32)
    if table_arr.size == 0:
        empty_values = np.full_like(score_batch["values"], SCORE_PADDING)
        return batch_with_values(score_batch, empty_values, padding_value=SCORE_PADDING)

    mapped = _apply_score_log_tail_table_numba(
        np.ascontiguousarray(score_batch["values"], dtype=np.float32),
        np.ascontiguousarray(score_batch["mask"], dtype=np.bool_),
        np.ascontiguousarray(table_arr[:, 0], dtype=np.float32),
        np.ascontiguousarray(table_arr[:, 1], dtype=np.float32),
        np.float32(SCORE_PADDING),
    )
    return batch_with_values(score_batch, mapped, padding_value=SCORE_PADDING)


def _build_length_mask(lengths: np.ndarray, width: int) -> np.ndarray:
    """Build one dense prefix mask from row lengths."""
    return np.arange(width, dtype=np.int64)[None, :] < np.asarray(lengths, dtype=np.int64)[:, None]


def apply_score_log_tail_table_to_profile_bundle(profile_bundle, table: np.ndarray):
    """Map one 3D profile bundle to empirical log-tail values using one lookup table."""
    table_arr = np.asarray(table, dtype=np.float32)
    values = np.ascontiguousarray(profile_bundle["values"], dtype=np.float32)
    lengths = np.asarray(profile_bundle["lengths"], dtype=np.int64)

    if table_arr.size == 0:
        empty_values = np.full_like(values, SCORE_PADDING)
        return pack_profile_bundle(empty_values, lengths, SCORE_PADDING)

    mask = np.ascontiguousarray(_build_length_mask(lengths, values.shape[2]), dtype=np.bool_)
    mapped = np.empty_like(values)
    scores_col = np.ascontiguousarray(table_arr[:, 0], dtype=np.float32)
    log_tail_col = np.ascontiguousarray(table_arr[:, 1], dtype=np.float32)

    for profile_index in range(values.shape[0]):
        mapped[profile_index] = _apply_score_log_tail_table_numba(
            values[profile_index],
            mask,
            scores_col,
            log_tail_col,
            np.float32(SCORE_PADDING),
        )

    return pack_profile_bundle(mapped, lengths, SCORE_PADDING)


def scores_to_empirical_log_tail_bundle(profile_bundle):
    """Convert one 3D profile bundle to empirical log-tail values within the current sample."""
    table = build_score_log_tail_table(flatten_profile_bundle(profile_bundle))
    return apply_score_log_tail_table_to_profile_bundle(profile_bundle, table)


def normalize_empirical_log_tail_pair(score_batch_plus, score_batch_minus):
    """Normalize two strand score batches using one shared empirical log-tail mapping."""
    profile_bundle = pack_profile_bundle(
        np.stack(
            (
                np.asarray(score_batch_plus["values"], dtype=np.float32),
                np.asarray(score_batch_minus["values"], dtype=np.float32),
            ),
            axis=0,
        ),
        np.asarray(score_batch_plus["lengths"], dtype=np.int64),
        SCORE_PADDING,
    )
    normalized = scores_to_empirical_log_tail_bundle(profile_bundle)
    values_plus = normalized["values"][0]
    values_minus = normalized["values"][1]
    return (
        pack_batch(values_plus, score_batch_plus["mask"], score_batch_plus["lengths"], SCORE_PADDING),
        pack_batch(values_minus, score_batch_minus["mask"], score_batch_minus["lengths"], SCORE_PADDING),
    )


def lookup_score_for_tail_probability(table: np.ndarray, tail_probability: float) -> float:
    """Convert a tail probability threshold to the corresponding score cutoff."""
    if tail_probability <= 0:
        return float(table[0, 0])

    target_log_tail = -np.log10(tail_probability)
    scores_col = table[:, 0]
    log_tail_col = table[:, 1]
    mask = log_tail_col >= target_log_tail

    if not np.any(mask):
        return float(scores_col[-1])

    last_valid = np.where(mask)[0][-1]
    return float(scores_col[last_valid])


def scores_to_empirical_log_tail(score_batch):
    """Convert one score batch to empirical log-tail values within the current sample."""
    table = build_score_log_tail_table(flatten_valid(score_batch))
    return apply_score_log_tail_table(score_batch, table)
