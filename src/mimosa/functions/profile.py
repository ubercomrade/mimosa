"""Profile bundle preparation and similarity metrics."""

from __future__ import annotations

import numpy as np
from numba import njit

from mimosa.batches import ProfileBundle, pack_profile_bundle


def prepare_profile_bundle(bundle: dict) -> ProfileBundle:
    """Return one contiguous profile bundle with explicit float32 values."""
    values = np.ascontiguousarray(bundle["values"], dtype=np.float32)
    lengths = np.ascontiguousarray(bundle["lengths"], dtype=np.int64)
    return pack_profile_bundle(values, lengths, bundle.get("padding_value", 0.0))


@njit(cache=False, nogil=True, fastmath=True)
def _overlap_sums_numba(values1: np.ndarray, values2: np.ndarray) -> tuple[float, float, float]:
    """Return sum(values1), sum(values2), and sum(min(values1, values2)) in one pass."""
    sum1 = 0.0
    sum2 = 0.0
    intersection = 0.0

    for index in range(values1.size):
        value1 = values1[index]
        value2 = values2[index]
        sum1 += value1
        sum2 += value2
        intersection += value1 if value1 < value2 else value2

    return sum1, sum2, intersection


def _flat_float32_pair(scores1: np.ndarray, scores2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous flattened float32 views for pairwise profile metrics."""
    values1 = np.ascontiguousarray(np.asarray(scores1, dtype=np.float32).ravel())
    values2 = np.ascontiguousarray(np.asarray(scores2, dtype=np.float32).ravel())
    if values1.size != values2.size:
        raise ValueError("scores1 and scores2 must contain the same number of values.")
    return values1, values2


def _window_float32_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous float32 window matrices with validated shape."""
    window_matrix_ndim = 2
    values_x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    values_y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    if values_x.shape != values_y.shape:
        raise ValueError("x and y must have the same shape.")
    if values_x.ndim != window_matrix_ndim:
        raise ValueError("x and y must be 2D arrays.")
    return values_x, values_y


def calc_co(scores1: np.ndarray, scores2: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the CO score over one selected window collection."""
    values1, values2 = _flat_float32_pair(scores1, scores2)
    sum1, sum2, intersection = _overlap_sums_numba(values1, values2)
    denom = min(sum1, sum2)
    if denom <= eps:
        return 0.0
    return float(intersection / denom)


def calc_dice(scores1: np.ndarray, scores2: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Dice score over one selected window collection."""
    values1, values2 = _flat_float32_pair(scores1, scores2)
    sum1, sum2, intersection = _overlap_sums_numba(values1, values2)
    denom = sum1 + sum2
    if denom <= eps:
        return 0.0
    return float((2.0 * intersection) / denom)


@njit(cache=False, nogil=True, fastmath=True)
def _rowwise_overlap_similarity_numba(
    values_x: np.ndarray,
    values_y: np.ndarray,
    eps: float,
    use_dice_denominator: bool,
) -> np.ndarray:
    """Compute one overlap-based similarity value per row without temporary arrays."""
    n_rows = values_x.shape[0]
    n_cols = values_x.shape[1]
    out = np.empty(n_rows, dtype=np.float32)

    for row_index in range(n_rows):
        sum_x = 0.0
        sum_y = 0.0
        intersection = 0.0

        for col_index in range(n_cols):
            value_x = values_x[row_index, col_index]
            value_y = values_y[row_index, col_index]
            sum_x += value_x
            sum_y += value_y
            intersection += value_x if value_x < value_y else value_y

        if use_dice_denominator:
            denom = sum_x + sum_y
            out[row_index] = (2.0 * intersection) / denom if denom > eps else np.nan
        else:
            denom = min(sum_x, sum_y)
            out[row_index] = intersection / denom if denom > eps else np.nan

    return out


def rowwise_co(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute one CO value per selected window."""
    values_x, values_y = _window_float32_pair(x, y)
    return _rowwise_overlap_similarity_numba(values_x, values_y, float(eps), False)


def rowwise_dice(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute one Dice value per selected window."""
    values_x, values_y = _window_float32_pair(x, y)
    return _rowwise_overlap_similarity_numba(values_x, values_y, float(eps), True)


@njit(cache=False, nogil=True, fastmath=True)
def _rowwise_cosine_numba(values_x: np.ndarray, values_y: np.ndarray, eps: float) -> np.ndarray:
    """Compute one cosine value per row without temporary arrays."""
    n_rows = values_x.shape[0]
    n_cols = values_x.shape[1]
    out = np.empty(n_rows, dtype=np.float32)

    for row_index in range(n_rows):
        dot = 0.0
        norm_x = 0.0
        norm_y = 0.0

        for col_index in range(n_cols):
            value_x = values_x[row_index, col_index]
            value_y = values_y[row_index, col_index]
            dot += value_x * value_y
            norm_x += value_x * value_x
            norm_y += value_y * value_y

        norm = np.sqrt(norm_x) * np.sqrt(norm_y)
        out[row_index] = dot / norm if norm > eps else np.nan

    return out


def rowwise_cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute one cosine value per selected window."""
    values_x, values_y = _window_float32_pair(x, y)
    return _rowwise_cosine_numba(values_x, values_y, float(eps))
