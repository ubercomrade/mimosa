# ruff: noqa: C901, PLR0912, PLR0913, PLR0915
"""Fused Numba kernels for profile-window alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

METRIC_CO = 0
METRIC_CO_ROWWISE = 1
METRIC_DICE = 2
METRIC_DICE_ROWWISE = 3
METRIC_COSINE = 4

METRIC_CODES = {
    "co": METRIC_CO,
    "co_rowwise": METRIC_CO_ROWWISE,
    "dice": METRIC_DICE,
    "dice_rowwise": METRIC_DICE_ROWWISE,
    "cosine": METRIC_COSINE,
}

# Below this amount of element work, prange overhead dominates on the benchmark matrix.
PROFILE_PARALLEL_WORK_THRESHOLD = 100_000
_EPS = 1e-6


@dataclass(slots=True)
class AlignmentWorkspace:
    """Reusable row-local storage for candidate deduplication and reductions."""

    marks: np.ndarray
    positions: np.ndarray
    partials: np.ndarray


def build_anchor_csr(rows: np.ndarray, positions: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert row/position anchors to stable CSR-like arrays."""
    row_array = np.ascontiguousarray(rows, dtype=np.int32)
    position_array = np.ascontiguousarray(positions, dtype=np.int32)
    if row_array.shape != position_array.shape:
        raise ValueError("Anchor rows and positions must have identical shapes.")
    if row_array.size and (int(row_array.min()) < 0 or int(row_array.max()) >= int(n_rows)):
        raise ValueError("Anchor row index is outside the profile bundle.")

    counts = np.bincount(row_array, minlength=int(n_rows))
    offsets = np.empty(int(n_rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    if row_array.size == 0 or np.all(row_array[1:] >= row_array[:-1]):
        return position_array, offsets
    order = np.argsort(row_array, kind="stable")
    return np.ascontiguousarray(position_array[order]), offsets


def make_alignment_workspace(n_rows: int, profile_width: int) -> AlignmentWorkspace:
    """Allocate reusable row-local candidate and reduction storage."""
    if int(n_rows) < 0 or int(profile_width) < 0:
        raise ValueError("Alignment workspace dimensions must be non-negative.")
    if int(n_rows) and int(profile_width) > np.iinfo(np.int64).max // int(n_rows):
        raise ValueError("Profile dimensions are too large for candidate key encoding.")
    shape = (int(n_rows), int(profile_width))
    return AlignmentWorkspace(
        marks=np.zeros(shape, dtype=np.int32),
        positions=np.empty(shape, dtype=np.int32),
        partials=np.empty((int(n_rows), 6), dtype=np.float64),
    )


def metric_code(metric: str) -> int:
    """Resolve a public profile metric name to its numerical kernel code."""
    try:
        return METRIC_CODES[str(metric)]
    except KeyError as exc:
        options = ", ".join(repr(name) for name in METRIC_CODES)
        raise ValueError(f"metric must be one of: {options}") from exc


@njit(cache=True, inline="always")
def _window_fits(position: int, length: int, radius: int) -> bool:
    return position - radius >= 0 and position + radius < length


@njit(cache=True, inline="always")
def _realign_query_position(scores, row: int, length: int, expected: int, radius: int) -> int:
    left = max(0, expected - radius)
    right = min(length - 1, expected + radius)
    if left > right:
        return -1
    best_position = left
    best_score = scores[row, left]
    for position in range(left + 1, right + 1):
        score = scores[row, position]
        if score > best_score:
            best_score = score
            best_position = position
    return best_position


@njit(cache=True, inline="always")
def _mark_candidate(marks, positions, row: int, position: int, generation: int, count: int) -> int:
    if marks[row, position] == generation:
        return count
    marks[row, position] = generation
    positions[row, count] = position
    return count + 1


@njit(cache=True)
def _collect_row_candidates(
    scores1,
    lengths1,
    lengths2,
    query_positions,
    query_offsets,
    target_positions,
    target_offsets,
    shift: int,
    window_radius: int,
    realign_window: int,
    marks,
    candidates,
    row: int,
    generation: int,
) -> int:
    """Collect unique query positions contributed by either anchor set."""
    length1 = int(lengths1[row])
    length2 = int(lengths2[row])
    count = 0
    for anchor_index in range(query_offsets[row], query_offsets[row + 1]):
        position1 = int(query_positions[anchor_index])
        position2 = position1 + shift
        if _window_fits(position1, length1, window_radius) and _window_fits(position2, length2, window_radius):
            count = _mark_candidate(marks, candidates, row, position1, generation, count)

    for anchor_index in range(target_offsets[row], target_offsets[row + 1]):
        expected_position1 = int(target_positions[anchor_index]) - shift
        position1 = _realign_query_position(scores1, row, length1, expected_position1, realign_window)
        if position1 < 0:
            continue
        position2 = position1 + shift
        if _window_fits(position1, length1, window_radius) and _window_fits(position2, length2, window_radius):
            count = _mark_candidate(marks, candidates, row, position1, generation, count)
    return count


@njit(cache=True)
def _accumulate_pooled_overlap(scores1, scores2, candidates, row: int, count: int, shift: int, radius: int):
    """Accumulate pooled overlap sums without materializing windows."""
    sum1 = 0.0
    sum2 = 0.0
    intersection = 0.0
    for candidate_index in range(count):
        position1 = int(candidates[row, candidate_index])
        position2 = position1 + shift
        for offset in range(-radius, radius + 1):
            value1 = scores1[row, position1 + offset]
            value2 = scores2[row, position2 + offset]
            sum1 += value1
            sum2 += value2
            intersection += value1 if value1 < value2 else value2
    return sum1, sum2, intersection


@njit(cache=True)
def _accumulate_rowwise_overlap(
    scores1, scores2, candidates, row: int, count: int, shift: int, radius: int, use_dice: bool
):
    """Accumulate finite per-window CO or Dice values."""
    score_sum = 0.0
    finite_count = 0
    for candidate_index in range(count):
        position1 = int(candidates[row, candidate_index])
        position2 = position1 + shift
        sum1 = 0.0
        sum2 = 0.0
        intersection = 0.0
        for offset in range(-radius, radius + 1):
            value1 = scores1[row, position1 + offset]
            value2 = scores2[row, position2 + offset]
            sum1 += value1
            sum2 += value2
            intersection += value1 if value1 < value2 else value2
        denominator = sum1 + sum2 if use_dice else min(sum1, sum2)
        if denominator > _EPS:
            score_sum += (2.0 * intersection if use_dice else intersection) / denominator
            finite_count += 1
    return score_sum, finite_count


@njit(cache=True)
def _accumulate_cosine(scores1, scores2, candidates, row: int, count: int, shift: int, radius: int):
    """Accumulate finite per-window cosine values."""
    score_sum = 0.0
    finite_count = 0
    for candidate_index in range(count):
        position1 = int(candidates[row, candidate_index])
        position2 = position1 + shift
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for offset in range(-radius, radius + 1):
            value1 = scores1[row, position1 + offset]
            value2 = scores2[row, position2 + offset]
            dot += value1 * value2
            norm1 += value1 * value1
            norm2 += value2 * value2
        denominator = np.sqrt(norm1) * np.sqrt(norm2)
        if denominator > _EPS:
            score_sum += dot / denominator
            finite_count += 1
    return score_sum, finite_count


@njit(cache=True)
def _score_alignment_row(
    scores1,
    lengths1,
    scores2,
    lengths2,
    query_positions,
    query_offsets,
    target_positions,
    target_offsets,
    shift: int,
    window_radius: int,
    realign_window: int,
    metric: int,
    marks,
    candidates,
    row: int,
    generation: int,
):
    """Collect, deduplicate, and score candidates for one independent row."""
    count = _collect_row_candidates(
        scores1,
        lengths1,
        lengths2,
        query_positions,
        query_offsets,
        target_positions,
        target_offsets,
        shift,
        window_radius,
        realign_window,
        marks,
        candidates,
        row,
        generation,
    )
    sum1 = 0.0
    sum2 = 0.0
    intersection = 0.0
    row_score_sum = 0.0
    finite_count = 0
    if metric in (METRIC_CO, METRIC_DICE):
        sum1, sum2, intersection = _accumulate_pooled_overlap(
            scores1, scores2, candidates, row, count, shift, window_radius
        )
    elif metric in (METRIC_CO_ROWWISE, METRIC_DICE_ROWWISE):
        row_score_sum, finite_count = _accumulate_rowwise_overlap(
            scores1,
            scores2,
            candidates,
            row,
            count,
            shift,
            window_radius,
            metric == METRIC_DICE_ROWWISE,
        )
    else:
        row_score_sum, finite_count = _accumulate_cosine(scores1, scores2, candidates, row, count, shift, window_radius)

    return sum1, sum2, intersection, row_score_sum, finite_count, count


@njit(cache=True, nogil=True)
def align_shift_serial(
    scores1,
    lengths1,
    scores2,
    lengths2,
    query_positions,
    query_offsets,
    target_positions,
    target_offsets,
    shift: int,
    window_radius: int,
    realign_window: int,
    metric: int,
    marks,
    candidates,
    partials,
    generation: int,
):
    """Evaluate one shift using a deterministic serial row loop."""
    for row in range(scores1.shape[0]):
        result = _score_alignment_row(
            scores1,
            lengths1,
            scores2,
            lengths2,
            query_positions,
            query_offsets,
            target_positions,
            target_offsets,
            shift,
            window_radius,
            realign_window,
            metric,
            marks,
            candidates,
            row,
            generation,
        )
        partials[row, 0] = result[0]
        partials[row, 1] = result[1]
        partials[row, 2] = result[2]
        partials[row, 3] = result[3]
        partials[row, 4] = result[4]
        partials[row, 5] = result[5]


@njit(cache=True, parallel=True, nogil=True)
def align_shift_parallel(
    scores1,
    lengths1,
    scores2,
    lengths2,
    query_positions,
    query_offsets,
    target_positions,
    target_offsets,
    shift: int,
    window_radius: int,
    realign_window: int,
    metric: int,
    marks,
    candidates,
    partials,
    generation: int,
):
    """Evaluate one shift with independent rows distributed by Numba."""
    for row in prange(scores1.shape[0]):
        result = _score_alignment_row(
            scores1,
            lengths1,
            scores2,
            lengths2,
            query_positions,
            query_offsets,
            target_positions,
            target_offsets,
            shift,
            window_radius,
            realign_window,
            metric,
            marks,
            candidates,
            row,
            generation,
        )
        partials[row, 0] = result[0]
        partials[row, 1] = result[1]
        partials[row, 2] = result[2]
        partials[row, 3] = result[3]
        partials[row, 4] = result[4]
        partials[row, 5] = result[5]


def score_shift(
    scores1: np.ndarray,
    lengths1: np.ndarray,
    scores2: np.ndarray,
    lengths2: np.ndarray,
    query_anchors: tuple[np.ndarray, np.ndarray],
    target_anchors: tuple[np.ndarray, np.ndarray],
    shift: int,
    window_radius: int,
    realign_window: int,
    metric: str,
    workspace: AlignmentWorkspace,
    generation: int,
    use_parallel: bool,
) -> tuple[float, int]:
    """Dispatch one fused shift calculation and reduce row-local partials."""
    code = metric_code(metric)
    kernel = align_shift_parallel if use_parallel else align_shift_serial
    kernel(
        scores1,
        lengths1,
        scores2,
        lengths2,
        query_anchors[0],
        query_anchors[1],
        target_anchors[0],
        target_anchors[1],
        int(shift),
        int(window_radius),
        int(realign_window),
        code,
        workspace.marks,
        workspace.positions,
        workspace.partials,
        int(generation),
    )

    partials = workspace.partials
    n_sites = int(np.sum(partials[:, 5], dtype=np.int64))
    if n_sites == 0:
        return 0.0, 0
    if code in (METRIC_CO, METRIC_DICE):
        sum1 = float(np.sum(partials[:, 0], dtype=np.float64))
        sum2 = float(np.sum(partials[:, 1], dtype=np.float64))
        intersection = float(np.sum(partials[:, 2], dtype=np.float64))
        denominator = min(sum1, sum2) if code == METRIC_CO else sum1 + sum2
        numerator = intersection if code == METRIC_CO else 2.0 * intersection
        return (numerator / denominator if denominator > _EPS else 0.0), n_sites

    finite_count = int(np.sum(partials[:, 4], dtype=np.int64))
    if finite_count == 0:
        return 0.0, n_sites
    return float(np.sum(partials[:, 3], dtype=np.float64) / finite_count), n_sites


def should_use_parallel(n_rows: int, profile_width: int, search_range: int, num_threads: int) -> bool:
    """Choose the parallel alignment path only when its measured overhead is amortized."""
    work = int(n_rows) * int(profile_width) * (2 * int(search_range) + 1)
    return int(num_threads) > 1 and work >= PROFILE_PARALLEL_WORK_THRESHOLD
