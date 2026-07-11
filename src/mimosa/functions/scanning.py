"""Numba-backed sequence scanning kernels."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from mimosa.batches import pack_batch

SCAN_PARALLEL_WORK_THRESHOLD = 1_000_000


def should_parallelize_scan(n_rows: int, max_scores: int, n_terms: int, kmer: int, n_strands: int = 1) -> bool:
    """Use row-parallel scanning only once its scheduling cost is amortized."""
    work = int(n_rows) * int(max_scores) * int(n_terms) * int(kmer) * int(n_strands)
    return get_num_threads() > 1 and work >= SCAN_PARALLEL_WORK_THRESHOLD


def score_seq(num_site, kmer, model):
    """Compute the score for one encoded site."""
    site = np.asarray(num_site, dtype=np.int64)
    matrix = np.asarray(model, dtype=np.float32).reshape(-1, np.asarray(model).shape[-1])
    kmer = int(kmer)
    score = 0.0

    for position in range(site.shape[0] - kmer + 1):
        code = 0
        for offset in range(kmer):
            code = code * 5 + int(site[position + offset])
        score += float(matrix[code, position])

    return score


def _prepare_model_rows(matrix: np.ndarray) -> np.ndarray:
    """Return one motif tensor as a flat 5-ary row table."""
    arr = np.asarray(matrix, dtype=np.float32)
    return np.ascontiguousarray(arr.reshape((-1, arr.shape[-1])), dtype=np.float32)


def _resolve_scan_layout(kmer: int, motif_len: int, with_context: bool) -> tuple[int, int, int]:
    """Resolve the geometry used by the sequence-scanning kernels."""
    context_len = kmer - 1 if with_context else 0
    window_size = motif_len + context_len
    n_terms = window_size - kmer + 1
    return context_len, window_size, n_terms


def _prepare_scan_inputs(sequences, matrix: np.ndarray):
    """Normalize scan inputs to contiguous arrays and derived geometry."""
    values = np.ascontiguousarray(sequences["values"], dtype=np.int8)
    lengths = np.ascontiguousarray(sequences["lengths"], dtype=np.int64)
    model_rows = _prepare_model_rows(matrix)
    motif_len = int(model_rows.shape[-1])
    out_lengths = np.maximum(lengths - motif_len + 1, 0)
    max_scores = int(out_lengths.max(initial=0))
    return values, lengths, model_rows, motif_len, max_scores, out_lengths


def _iter_scan_buckets(lengths: np.ndarray, motif_len: int, bucket_step: int):
    """Yield row-index buckets with similar output lengths."""
    out_lengths = np.maximum(lengths - motif_len + 1, 0)
    positive_indices = np.flatnonzero(out_lengths > 0)
    if positive_indices.size == 0:
        return

    bucket_ids = (out_lengths[positive_indices] - 1) // max(int(bucket_step), 1)
    order = np.argsort(bucket_ids, kind="mergesort")
    sorted_indices = positive_indices[order]
    sorted_bucket_ids = bucket_ids[order]

    starts = np.r_[0, np.flatnonzero(np.diff(sorted_bucket_ids)) + 1]
    stops = np.r_[starts[1:], sorted_indices.size]
    for start, stop in zip(starts, stops, strict=False):
        yield sorted_indices[start:stop]


@njit(cache=False, fastmath=True, nogil=False)
def _score_window_forward(seq_row, length: int, model_rows, pos: int, kmer: int, context_len: int, n_terms: int):
    """Score one forward-aligned window."""
    total = np.float32(0.0)
    for term in range(n_terms):
        code = 0
        src_start = pos - context_len + term
        for offset in range(kmer):
            src = src_start + offset
            encoded = 4
            if 0 <= src < length:
                encoded = int(seq_row[src])
            code = code * 5 + encoded
        total += model_rows[code, term]
    return total


@njit(cache=False, fastmath=True, nogil=False)
def _score_window_reverse(seq_row, length: int, model_rows, pos: int, kmer: int, window_size: int, n_terms: int):
    """Score one reverse-complement-aligned window."""
    complement_offset = 3
    nucleotide_padding = 4
    total = np.float32(0.0)
    for term in range(n_terms):
        code = 0
        for offset in range(kmer):
            src = pos + (window_size - 1 - (term + offset))
            encoded = nucleotide_padding
            if 0 <= src < length:
                base = int(seq_row[src])
                encoded = nucleotide_padding if base == nucleotide_padding else complement_offset - base
            code = code * 5 + encoded
        total += model_rows[code, term]
    return total


@njit(cache=False, fastmath=True, nogil=False)
def _scan_dense_kernel_numba(values, lengths, model_rows, kmer: int, context_len: int, n_terms: int):
    """Score one dense encoded sequence batch for one strand."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_forward(
                seq_row,
                length,
                model_rows,
                pos,
                kmer,
                context_len,
                n_terms,
            )
            mask[row_index, pos] = True

    return scores, mask


@njit(cache=False, parallel=True, fastmath=True, nogil=False)
def _scan_dense_kernel_parallel_numba(values, lengths, model_rows, kmer: int, context_len: int, n_terms: int):
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)
    for row_index in prange(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_forward(seq_row, length, model_rows, pos, kmer, context_len, n_terms)
            mask[row_index, pos] = True
    return scores, mask


@njit(cache=False, fastmath=True, nogil=False)
def _scan_dense_reverse_kernel_numba(values, lengths, model_rows, kmer: int, window_size: int, n_terms: int):
    """Score one dense encoded sequence batch on the reverse-complement strand."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_reverse(seq_row, length, model_rows, pos, kmer, window_size, n_terms)
            mask[row_index, pos] = True

    return scores, mask


@njit(cache=False, parallel=True, fastmath=True, nogil=False)
def _scan_dense_reverse_kernel_parallel_numba(values, lengths, model_rows, kmer: int, window_size: int, n_terms: int):
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, max_scores), dtype=np.bool_)
    for row_index in prange(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, pos] = _score_window_reverse(seq_row, length, model_rows, pos, kmer, window_size, n_terms)
            mask[row_index, pos] = True
    return scores, mask


@njit(cache=False, fastmath=True, nogil=False)
def _scan_dense_strands_kernel_numba(
    values, lengths, model_rows, kmer: int, context_len: int, window_size: int, n_terms: int
):
    """Score one dense encoded batch on both strands in one call."""
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, 2, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, 2, max_scores), dtype=np.bool_)

    for row_index in range(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        if n_scores == 0:
            continue

        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, 0, pos] = _score_window_forward(
                seq_row,
                length,
                model_rows,
                pos,
                kmer,
                context_len,
                n_terms,
            )
            scores[row_index, 1, pos] = _score_window_reverse(
                seq_row,
                length,
                model_rows,
                pos,
                kmer,
                window_size,
                n_terms,
            )
            mask[row_index, 0, pos] = True
            mask[row_index, 1, pos] = True

    return scores, mask


@njit(cache=False, parallel=True, fastmath=True, nogil=False)
def _scan_dense_strands_kernel_parallel_numba(
    values, lengths, model_rows, kmer: int, context_len: int, window_size: int, n_terms: int
):
    n_rows, _ = values.shape
    motif_len = model_rows.shape[-1]
    max_scores = max(values.shape[1] - motif_len + 1, 0)
    scores = np.zeros((n_rows, 2, max_scores), dtype=np.float32)
    mask = np.zeros((n_rows, 2, max_scores), dtype=np.bool_)
    for row_index in prange(n_rows):
        length = int(lengths[row_index])
        n_scores = max(length - motif_len + 1, 0)
        seq_row = values[row_index]
        for pos in range(n_scores):
            scores[row_index, 0, pos] = _score_window_forward(
                seq_row, length, model_rows, pos, kmer, context_len, n_terms
            )
            scores[row_index, 1, pos] = _score_window_reverse(
                seq_row, length, model_rows, pos, kmer, window_size, n_terms
            )
            mask[row_index, 0, pos] = True
            mask[row_index, 1, pos] = True
    return scores, mask


def _scan_one_kernel(values, lengths, model_rows, kmer, context_len, window_size, n_terms, is_revcomp):
    max_scores = max(values.shape[1] - model_rows.shape[-1] + 1, 0)
    parallel = should_parallelize_scan(values.shape[0], max_scores, n_terms, kmer)
    if is_revcomp:
        kernel = _scan_dense_reverse_kernel_parallel_numba if parallel else _scan_dense_reverse_kernel_numba
        return kernel(values, lengths, model_rows, kmer, window_size, n_terms)
    kernel = _scan_dense_kernel_parallel_numba if parallel else _scan_dense_kernel_numba
    return kernel(values, lengths, model_rows, kmer, context_len, n_terms)


def _scan_strands_kernel(values, lengths, model_rows, kmer, context_len, window_size, n_terms):
    parallel = should_parallelize_scan(
        values.shape[0], max(values.shape[1] - model_rows.shape[-1] + 1, 0), n_terms, kmer, 2
    )
    kernel = _scan_dense_strands_kernel_parallel_numba if parallel else _scan_dense_strands_kernel_numba
    return kernel(values, lengths, model_rows, kmer, context_len, window_size, n_terms)


def _empty_score_scan_batch(n_rows: int, max_scores: int, out_lengths: np.ndarray):
    """Return one empty score batch with the requested output geometry."""
    score_padding = 0.0
    empty_values = np.full((n_rows, max_scores), score_padding, dtype=np.float32)
    empty_mask = np.zeros((n_rows, max_scores), dtype=bool)
    return pack_batch(empty_values, empty_mask, out_lengths, score_padding)


def batch_all_scores(
    sequences, matrix: np.ndarray, kmer: int = 1, is_revcomp: bool = False, with_context: bool = False
):
    """Compute scores for all sequences in one dense masked batch."""
    values, lengths, model_rows, motif_len, max_scores, out_lengths = _prepare_scan_inputs(sequences, matrix)
    n_rows = int(values.shape[0])

    if n_rows == 0 or max_scores == 0:
        return _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    context_len, window_size, n_terms = _resolve_scan_layout(int(kmer), motif_len, bool(with_context))
    if int(lengths.max(initial=0)) == int(lengths.min(initial=0)):
        scored_values, scored_mask = _scan_one_kernel(
            values, lengths, model_rows, int(kmer), context_len, window_size, n_terms, is_revcomp
        )
    else:
        score_padding = 0.0
        bucket_step = 32
        scored_values = np.full((n_rows, max_scores), score_padding, dtype=np.float32)
        scored_mask = np.zeros((n_rows, max_scores), dtype=np.bool_)
        for bucket_indices in _iter_scan_buckets(lengths, motif_len, bucket_step):
            bucket_lengths = np.ascontiguousarray(lengths[bucket_indices], dtype=np.int64)
            bucket_width = int(bucket_lengths.max(initial=0))
            bucket_values = np.ascontiguousarray(values[bucket_indices, :bucket_width], dtype=np.int8)
            bucket_scores, bucket_mask = _scan_one_kernel(
                bucket_values, bucket_lengths, model_rows, int(kmer), context_len, window_size, n_terms, is_revcomp
            )
            bucket_score_width = bucket_scores.shape[1]
            scored_values[bucket_indices, :bucket_score_width] = bucket_scores
            scored_mask[bucket_indices, :bucket_score_width] = bucket_mask

    return pack_batch(scored_values, scored_mask, out_lengths, 0.0)


def batch_all_scores_strands(sequences, matrix: np.ndarray, kmer: int = 1, with_context: bool = False):
    """Compute scores for both strands in one dense masked batch call."""
    values, lengths, model_rows, motif_len, max_scores, out_lengths = _prepare_scan_inputs(sequences, matrix)
    n_rows = int(values.shape[0])

    if n_rows == 0 or max_scores == 0:
        empty_batch = _empty_score_scan_batch(n_rows, max_scores, out_lengths)
        return empty_batch, _empty_score_scan_batch(n_rows, max_scores, out_lengths)

    context_len, window_size, n_terms = _resolve_scan_layout(int(kmer), motif_len, bool(with_context))
    if int(lengths.max(initial=0)) == int(lengths.min(initial=0)):
        scored_values, scored_mask = _scan_strands_kernel(
            values, lengths, model_rows, int(kmer), context_len, window_size, n_terms
        )
    else:
        score_padding = 0.0
        bucket_step = 32
        scored_values = np.full((n_rows, 2, max_scores), score_padding, dtype=np.float32)
        scored_mask = np.zeros((n_rows, 2, max_scores), dtype=np.bool_)
        for bucket_indices in _iter_scan_buckets(lengths, motif_len, bucket_step):
            bucket_lengths = np.ascontiguousarray(lengths[bucket_indices], dtype=np.int64)
            bucket_width = int(bucket_lengths.max(initial=0))
            bucket_values = np.ascontiguousarray(values[bucket_indices, :bucket_width], dtype=np.int8)
            bucket_scores, bucket_mask = _scan_strands_kernel(
                bucket_values, bucket_lengths, model_rows, int(kmer), context_len, window_size, n_terms
            )
            bucket_score_width = bucket_scores.shape[2]
            scored_values[bucket_indices, :, :bucket_score_width] = bucket_scores
            scored_mask[bucket_indices, :, :bucket_score_width] = bucket_mask

    plus_batch = pack_batch(scored_values[:, 0, :], scored_mask[:, 0, :], out_lengths, 0.0)
    minus_batch = pack_batch(scored_values[:, 1, :], scored_mask[:, 1, :], out_lengths, 0.0)
    return plus_batch, minus_batch
