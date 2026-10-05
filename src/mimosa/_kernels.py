"""Numba kernels: scanning, normalization, anchors, and weighted Jaccard.

Primitive arguments only; never imports public API.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

N_CODE = 4


@njit(cache=True)
def pwm_scan_forward(weights, seq, n_pos, n_terms, out):
    for pos in range(n_pos):
        total = np.float32(0.0)
        for term in range(n_terms):
            total += weights[seq[pos + term], term]
        out[pos] = total


@njit(cache=True)
def pwm_scan_reverse(weights, seq, n_pos, n_terms, out):
    for pos in range(n_pos):
        total = np.float32(0.0)
        for term in range(n_terms):
            base = seq[pos + n_terms - 1 - term]
            if base != N_CODE:
                base = 3 - base
            total += weights[base, term]
        out[pos] = total


@njit(cache=True)
def _ho_oriented_base(seq, index, length, reverse_complement):
    if index < 0 or index >= length:
        return N_CODE
    sequence_index = length - index - 1 if reverse_complement else index
    base = seq[sequence_index]
    if reverse_complement and base != N_CODE:
        base = 3 - base
    return base


@njit(cache=True)
def ho_kmer_codes(seq, kmer_size, first_start, n_codes, reverse_complement, out):
    length = seq.shape[0]
    code = 0
    for offset in range(kmer_size):
        code = 5 * code + _ho_oriented_base(
            seq, first_start + offset, length, reverse_complement
        )
    out[0] = code
    leading_weight = 5 ** (kmer_size - 1)
    for i in range(1, n_codes):
        start = first_start + i - 1
        code = (
            5
            * (code - _ho_oriented_base(seq, start, length, reverse_complement) * leading_weight)
            + _ho_oriented_base(seq, start + kmer_size, length, reverse_complement)
        )
        out[i] = code


@njit(cache=True)
def rolling_scan_forward(weights, seq, kmer_size, n_terms, n_pos, out):
    n_codes = n_pos + n_terms - 1
    # The largest supported context has kmer_size=11, so 5**11 - 1 fits in
    # Int32. Halving this per-row scratch buffer matters for high-order scans.
    codes = np.empty(n_codes, dtype=np.int32)
    ho_kmer_codes(seq, kmer_size, 0, n_codes, False, codes)
    for pos in range(n_pos):
        total = np.float32(0.0)
        for term in range(n_terms):
            total += weights[codes[pos + term], term]
        out[pos] = total


@njit(cache=True)
def rolling_scan_reverse(weights, seq, kmer_size, n_terms, n_pos, out):
    n_codes = n_pos + n_terms - 1
    codes = np.empty(n_codes, dtype=np.int32)
    ho_kmer_codes(seq, kmer_size, 0, n_codes, True, codes)
    for pos in range(n_pos):
        total = np.float32(0.0)
        for term in range(n_terms):
            total += weights[codes[n_pos - pos - 1 + term], term]
        out[pos] = total


@njit(cache=True)
def batch_scan_forward(weights, seq_data, seq_offsets, out_data, out_offsets, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in range(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        pwm_scan_forward(weights, seq_data[start:stop], n_pos, n_terms, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(parallel=True, cache=True)
def batch_scan_forward_parallel(weights, seq_data, seq_offsets, out_data, out_offsets, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in prange(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        pwm_scan_forward(weights, seq_data[start:stop], n_pos, n_terms, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(cache=True)
def batch_scan_reverse(weights, seq_data, seq_offsets, out_data, out_offsets, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in range(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        pwm_scan_reverse(weights, seq_data[start:stop], n_pos, n_terms, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(parallel=True, cache=True)
def batch_scan_reverse_parallel(weights, seq_data, seq_offsets, out_data, out_offsets, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in prange(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        pwm_scan_reverse(weights, seq_data[start:stop], n_pos, n_terms, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(cache=True)
def batch_rolling_forward(weights, seq_data, seq_offsets, out_data, out_offsets, kmer_size, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in range(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        rolling_scan_forward(weights, seq_data[start:stop], kmer_size, n_terms, n_pos, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(parallel=True, cache=True)
def batch_rolling_forward_parallel(weights, seq_data, seq_offsets, out_data, out_offsets, kmer_size, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in prange(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        rolling_scan_forward(weights, seq_data[start:stop], kmer_size, n_terms, n_pos, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(cache=True)
def batch_rolling_reverse(weights, seq_data, seq_offsets, out_data, out_offsets, kmer_size, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in range(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        rolling_scan_reverse(weights, seq_data[start:stop], kmer_size, n_terms, n_pos, out_data[out_offsets[row] : out_offsets[row + 1]])


@njit(parallel=True, cache=True)
def batch_rolling_reverse_parallel(weights, seq_data, seq_offsets, out_data, out_offsets, kmer_size, n_terms):
    n_rows = seq_offsets.shape[0] - 1
    for row in prange(n_rows):
        start = seq_offsets[row]
        stop = seq_offsets[row + 1]
        n_pos = out_offsets[row + 1] - out_offsets[row]
        if n_pos <= 0:
            continue
        rolling_scan_reverse(weights, seq_data[start:stop], kmer_size, n_terms, n_pos, out_data[out_offsets[row] : out_offsets[row + 1]])



# ── Circular padding ─────────────────────────────────────────────────────────

@njit(cache=True)
def circular_pad_rows(seq_data, seq_offsets, left, right, out_data, out_offsets):
    """Wrap every non-empty row so a linear scan yields one score per base."""
    n_rows = seq_offsets.shape[0] - 1
    cursor = 0
    out_offsets[0] = 0
    for row in range(n_rows):
        start = seq_offsets[row]
        length = seq_offsets[row + 1] - start
        if length > 0:
            source = (-left) % length
            for _ in range(length + left + right):
                out_data[cursor] = seq_data[start + source]
                cursor += 1
                source += 1
                if source == length:
                    source = 0
        out_offsets[row + 1] = cursor


# ── Profile normalization ────────────────────────────────────────────────────

@njit(cache=True)
def _upper_bound_desc(scores, target):
    """Return the last descending-table score that is at least ``target``."""
    n = scores.shape[0]
    if n <= 1:
        return 0
    if target >= scores[0]:
        return 0
    if target <= scores[n - 1]:
        return n - 1
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if scores[mid] >= target:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@njit(cache=True)
def transform_empirical_scores(scores, table_scores, table_log_tail, out):
    for i in range(scores.shape[0]):
        out[i] = table_log_tail[_upper_bound_desc(table_scores, scores[i])]


@njit(parallel=True, cache=True)
def transform_empirical_scores_parallel(scores, table_scores, table_log_tail, out):
    for i in prange(scores.shape[0]):
        out[i] = table_log_tail[_upper_bound_desc(table_scores, scores[i])]


@njit(cache=True)
def _transform_hybrid_one(
    score, minimum, bin_width, histogram_log_tail, exact_scores, exact_log_tail
):
    n_bins = histogram_log_tail.shape[0]
    exact_size = exact_scores.shape[0]
    if exact_size > 0 and score >= exact_scores[exact_size - 1]:
        return exact_log_tail[_upper_bound_desc(exact_scores, score)]
    if n_bins == 0:
        return np.float32(0.0)
    if bin_width == 0.0:
        index = 0
    else:
        index = int(np.floor((float(score) - float(minimum)) / bin_width))
    if index < 0:
        index = 0
    elif index >= n_bins:
        index = n_bins - 1
    return histogram_log_tail[index]


@njit(cache=True)
def transform_hybrid_scores(
    scores, minimum, bin_width, histogram_log_tail, exact_scores, exact_log_tail, out
):
    for i in range(scores.shape[0]):
        out[i] = _transform_hybrid_one(
            scores[i], minimum, bin_width, histogram_log_tail, exact_scores, exact_log_tail
        )


@njit(parallel=True, cache=True)
def transform_hybrid_scores_parallel(
    scores, minimum, bin_width, histogram_log_tail, exact_scores, exact_log_tail, out
):
    for i in prange(scores.shape[0]):
        out[i] = _transform_hybrid_one(
            scores[i], minimum, bin_width, histogram_log_tail, exact_scores, exact_log_tail
        )


# ── Anchor collection ────────────────────────────────────────────────────────

@njit(cache=True)
def count_unique_sorted(values):
    if values.shape[0] == 0:
        return 0
    count = 1
    for i in range(1, values.shape[0]):
        if values[i] != values[i - 1]:
            count += 1
    return count


@njit(cache=True)
def fill_empirical_table_sorted(values, total_n, scores_out, log_tail_out):
    out = 0
    cumulative = 0
    i = values.shape[0] - 1
    while i >= 0:
        score = values[i]
        stop = i
        while i >= 0 and values[i] == score:
            i -= 1
        cumulative += stop - i
        scores_out[out] = score
        log_tail_out[out] = np.float32(-np.log10(cumulative / total_n))
        out += 1


@njit(cache=True)
def collect_threshold_anchors_csr(
    scores_data, scores_offsets, threshold, positions_out, anchor_offsets_out
):
    n_rows = scores_offsets.shape[0] - 1
    count = 0
    anchor_offsets_out[0] = 0
    for row in range(n_rows):
        start = scores_offsets[row]
        stop = scores_offsets[row + 1]
        for j in range(start, stop):
            if scores_data[j] >= threshold:
                positions_out[count] = j - start
                count += 1
        anchor_offsets_out[row + 1] = count
    return count


@njit(cache=True)
def count_threshold_anchors_csr(
    scores_data, scores_offsets, threshold, anchor_offsets_out
):
    n_rows = scores_offsets.shape[0] - 1
    count = 0
    anchor_offsets_out[0] = 0
    for row in range(n_rows):
        start = scores_offsets[row]
        stop = scores_offsets[row + 1]
        for j in range(start, stop):
            if scores_data[j] >= threshold:
                count += 1
        anchor_offsets_out[row + 1] = count
    return count


# ── Weighted Jaccard kernels ─────────────────────────────────────────────────

@njit(cache=True)
def _jaccard_row(
    q_data, t_data, row_start, length,
    q_pos, q_lo, q_hi,
    t_pos, t_lo, t_hi,
    shift, tau0,
):
    """Sum min/max over one circular row where either profile reaches ``tau0``.

    Query position ``i`` is paired with target position ``(i + shift) mod length``.
    """
    min_sum = 0.0
    max_sum = 0.0
    n = 0
    if length <= 0:
        return min_sum, max_sum, n
    step = shift % length
    for idx in range(q_lo, q_hi):
        pos = q_pos[idx]
        partner = pos + step
        if partner >= length:
            partner -= length
        u = q_data[row_start + pos]
        v = t_data[row_start + partner]
        if u < v:
            min_sum += u
            max_sum += v
        else:
            min_sum += v
            max_sum += u
        n += 1
    for idx in range(t_lo, t_hi):
        pos = t_pos[idx]
        partner = pos - step
        if partner < 0:
            partner += length
        u = q_data[row_start + partner]
        # Query anchors were already paired in the first pass.
        if u >= tau0:
            continue
        v = t_data[row_start + pos]
        if u < v:
            min_sum += u
            max_sum += v
        else:
            min_sum += v
            max_sum += u
        n += 1
    return min_sum, max_sum, n


@njit(cache=True)
def _jaccard_orientation(
    q_data, t_data, offsets,
    q_pos, q_offsets,
    t_pos, t_offsets,
    search_range, tau0,
    out_scores, out_sites,
):
    n_rows = offsets.shape[0] - 1
    n_shifts = out_scores.shape[0]
    min_sums = np.zeros(n_shifts, dtype=np.float64)
    max_sums = np.zeros(n_shifts, dtype=np.float64)
    out_sites[:] = 0
    for row in range(n_rows):
        # Most rows hold no anchor of either profile at the usual thresholds.
        if q_offsets[row] == q_offsets[row + 1] and t_offsets[row] == t_offsets[row + 1]:
            continue
        for shift_index in range(n_shifts):
            min_sum, max_sum, sites = _jaccard_row(
                q_data, t_data, offsets[row], offsets[row + 1] - offsets[row],
                q_pos, q_offsets[row], q_offsets[row + 1],
                t_pos, t_offsets[row], t_offsets[row + 1],
                shift_index - search_range, tau0,
            )
            min_sums[shift_index] += min_sum
            max_sums[shift_index] += max_sum
            out_sites[shift_index] += sites
    for shift_index in range(n_shifts):
        max_sum = max_sums[shift_index]
        out_scores[shift_index] = min_sums[shift_index] / max_sum if max_sum > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _jaccard_orientation_parallel(
    q_data, t_data, offsets,
    q_pos, q_offsets,
    t_pos, t_offsets,
    search_range, tau0,
    row_min, row_max, row_sites,
    out_scores, out_sites,
):
    n_rows = offsets.shape[0] - 1
    n_shifts = out_scores.shape[0]
    for row in prange(n_rows):
        if q_offsets[row] == q_offsets[row + 1] and t_offsets[row] == t_offsets[row + 1]:
            for shift_index in range(n_shifts):
                row_min[row, shift_index] = 0.0
                row_max[row, shift_index] = 0.0
                row_sites[row, shift_index] = 0
            continue
        for shift_index in range(n_shifts):
            min_sum, max_sum, sites = _jaccard_row(
                q_data, t_data, offsets[row], offsets[row + 1] - offsets[row],
                q_pos, q_offsets[row], q_offsets[row + 1],
                t_pos, t_offsets[row], t_offsets[row + 1],
                shift_index - search_range, tau0,
            )
            row_min[row, shift_index] = min_sum
            row_max[row, shift_index] = max_sum
            row_sites[row, shift_index] = sites

    # Rows are reduced in order, exactly as the serial kernel accumulates them,
    # so both kernels agree bit for bit regardless of the thread count.
    for shift_index in range(n_shifts):
        min_sum = 0.0
        max_sum = 0.0
        sites = 0
        for row in range(n_rows):
            min_sum += row_min[row, shift_index]
            max_sum += row_max[row, shift_index]
            sites += row_sites[row, shift_index]
        out_scores[shift_index] = min_sum / max_sum if max_sum > 0.0 else 0.0
        out_sites[shift_index] = sites
