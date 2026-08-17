"""Numba kernels: scanning, normalization, anchors, and alignment.

Primitive arguments only; never imports public API.
"""

from __future__ import annotations

import numpy as np
from numba import get_thread_id, njit, prange

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
def collect_best_anchors_csr(scores_data, scores_offsets, positions_out, anchor_offsets_out):
    n_rows = scores_offsets.shape[0] - 1
    count = 0
    anchor_offsets_out[0] = 0
    for row in range(n_rows):
        start = scores_offsets[row]
        stop = scores_offsets[row + 1]
        if start != stop:
            best_pos = start
            best_score = scores_data[start]
            for j in range(start + 1, stop):
                if scores_data[j] > best_score:
                    best_score = scores_data[j]
                    best_pos = j
            positions_out[count] = best_pos - start
            count += 1
        anchor_offsets_out[row + 1] = count
    return count


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


# ── Alignment kernels ─────────────────────────────────────────────────────────

@njit(cache=True)
def _window_fits(pos, length, radius):
    return pos - radius >= 0 and pos + radius < length


@njit(cache=True)
def _realign_query_position(r, expected, radius):
    length = r.shape[0]
    left = expected - radius
    if left < 0:
        left = 0
    right = expected + radius
    if right >= length:
        right = length - 1
    if left > right:
        return -1
    best_pos = left
    best_score = r[left]
    for pos in range(left + 1, right + 1):
        if r[pos] > best_score:
            best_score = r[pos]
            best_pos = pos
    return best_pos


@njit(cache=True)
def _accumulate_overlap(r1, r2, pos1, shift, window_radius, use_dice):
    pos2 = pos1 + shift
    sum1 = 0.0
    sum2 = 0.0
    intersection = 0.0
    for offset in range(-window_radius, window_radius + 1):
        v1 = r1[pos1 + offset]
        v2 = r2[pos2 + offset]
        sum1 += v1
        sum2 += v2
        if v1 < v2:
            intersection += v1
        else:
            intersection += v2
    denom = sum1 + sum2 if use_dice else (sum1 if sum1 < sum2 else sum2)
    if denom > 1e-6:
        return (2.0 * intersection / denom if use_dice else intersection / denom), 1
    return 0.0, 0


@njit(cache=True)
def _accumulate_cosine(r1, r2, pos1, shift, window_radius):
    pos2 = pos1 + shift
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for offset in range(-window_radius, window_radius + 1):
        v1 = r1[pos1 + offset]
        v2 = r2[pos2 + offset]
        dot += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2
    denom = np.sqrt(norm1) * np.sqrt(norm2)
    if denom > 1e-6:
        return dot / denom, 1
    return 0.0, 0


@njit(cache=True)
def _score_row_csr(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    query_site_start_offset, target_site_start_offset,
    row, shift, window_radius, realign_window,
    metric_kind, use_dice, seen, epoch,
):
    len1 = scores1_offsets[row + 1] - scores1_offsets[row]
    len2 = scores2_offsets[row + 1] - scores2_offsets[row]
    r1_start = scores1_offsets[row]
    r2_start = scores2_offsets[row]
    r1 = scores1_data[r1_start : r1_start + len1]
    r2 = scores2_data[r2_start : r2_start + len2]
    scan_shift = shift + query_site_start_offset - target_site_start_offset
    total_row_score = 0.0
    total_finite = 0
    total_sites = 0

    for idx in range(query_offsets[row], query_offsets[row + 1]):
        pos1 = query_positions[idx] - query_site_start_offset
        pos2 = pos1 + scan_shift
        if _window_fits(pos1, len1, window_radius) and _window_fits(
            pos2, len2, window_radius
        ):
            if seen[pos1] != epoch:
                seen[pos1] = epoch
                if metric_kind == 1:
                    score, finite = _accumulate_cosine(
                        r1, r2, pos1, scan_shift, window_radius
                    )
                else:
                    score, finite = _accumulate_overlap(
                        r1, r2, pos1, scan_shift, window_radius, use_dice
                    )
                total_row_score += score
                total_finite += finite
                total_sites += finite

    for idx in range(target_offsets[row], target_offsets[row + 1]):
        expected_pos1 = target_positions[idx] - shift - query_site_start_offset
        pos1 = _realign_query_position(r1, expected_pos1, realign_window)
        if pos1 < 0:
            continue
        pos2 = pos1 + scan_shift
        if _window_fits(pos1, len1, window_radius) and _window_fits(
            pos2, len2, window_radius
        ):
            if seen[pos1] != epoch:
                seen[pos1] = epoch
                if metric_kind == 1:
                    score, finite = _accumulate_cosine(
                        r1, r2, pos1, scan_shift, window_radius
                    )
                else:
                    score, finite = _accumulate_overlap(
                        r1, r2, pos1, scan_shift, window_radius, use_dice
                    )
                total_row_score += score
                total_finite += finite
                total_sites += finite
    return total_row_score, total_finite, total_sites


@njit(cache=True)
def _score_row_best(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    query_site_start_offset, target_site_start_offset,
    row, shift, window_radius, realign_window,
    metric_kind, use_dice,
):
    len1 = scores1_offsets[row + 1] - scores1_offsets[row]
    len2 = scores2_offsets[row + 1] - scores2_offsets[row]
    r1_start = scores1_offsets[row]
    r2_start = scores2_offsets[row]
    r1 = scores1_data[r1_start : r1_start + len1]
    r2 = scores2_data[r2_start : r2_start + len2]
    scan_shift = shift + query_site_start_offset - target_site_start_offset
    query_pos = -1
    target_pos = -1

    if query_offsets[row] < query_offsets[row + 1]:
        candidate = query_positions[query_offsets[row]] - query_site_start_offset
        if _window_fits(candidate, len1, window_radius) and _window_fits(
            candidate + scan_shift, len2, window_radius
        ):
            query_pos = candidate

    if target_offsets[row] < target_offsets[row + 1]:
        expected = target_positions[target_offsets[row]] - shift - query_site_start_offset
        candidate = _realign_query_position(r1, expected, realign_window)
        if candidate >= 0 and _window_fits(
            candidate, len1, window_radius
        ) and _window_fits(candidate + scan_shift, len2, window_radius):
            target_pos = candidate

    total_row_score = 0.0
    total_finite = 0
    total_sites = 0
    if query_pos >= 0:
        if metric_kind == 1:
            score, finite = _accumulate_cosine(
                r1, r2, query_pos, scan_shift, window_radius
            )
        else:
            score, finite = _accumulate_overlap(
                r1, r2, query_pos, scan_shift, window_radius, use_dice
            )
        total_row_score += score
        total_finite += finite
        total_sites += finite

    if target_pos >= 0 and target_pos != query_pos:
        if metric_kind == 1:
            score, finite = _accumulate_cosine(
                r1, r2, target_pos, scan_shift, window_radius
            )
        else:
            score, finite = _accumulate_overlap(
                r1, r2, target_pos, scan_shift, window_radius, use_dice
            )
        total_row_score += score
        total_finite += finite
        total_sites += finite
    return total_row_score, total_finite, total_sites


@njit(cache=True)
def _score_shift_csr(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    query_site_start_offset, target_site_start_offset,
    shift, window_radius, realign_window,
    metric_kind,  # 0=rowwise, 1=cosine
    use_dice, seen,
    epoch_base,
    out_score, out_sites,
):
    total_row_score = 0.0
    total_finite = 0
    total_sites = 0
    for row in range(scores1_offsets.shape[0] - 1):
        row_score, row_finite, row_sites = _score_row_csr(
            scores1_data, scores1_offsets,
            scores2_data, scores2_offsets,
            query_positions, query_offsets,
            target_positions, target_offsets,
            query_site_start_offset, target_site_start_offset,
            row, shift, window_radius, realign_window,
            metric_kind, use_dice, seen, epoch_base + row + 1,
        )
        total_row_score += row_score
        total_finite += row_finite
        total_sites += row_sites
    out_score[0] = 0.0 if total_sites == 0 or total_finite == 0 else total_row_score / total_finite
    out_sites[0] = total_sites


@njit(parallel=True, cache=True)
def _score_orientation_csr_parallel(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    query_site_start_offset, target_site_start_offset,
    search_range, window_radius, realign_window,
    metric_kind, use_dice, seen,
    row_scores, row_finite, row_sites, out_scores, out_sites,
):
    n_rows = scores1_offsets.shape[0] - 1
    n_shifts = 2 * search_range + 1
    for row in prange(n_rows):
        thread = get_thread_id()
        for shift_index in range(n_shifts):
            shift = shift_index - search_range
            score, finite, sites = _score_row_csr(
                scores1_data, scores1_offsets,
                scores2_data, scores2_offsets,
                query_positions, query_offsets,
                target_positions, target_offsets,
                query_site_start_offset, target_site_start_offset,
                row, shift, window_radius, realign_window,
                metric_kind, use_dice, seen[thread], shift_index * n_rows + row + 1,
            )
            row_scores[row, shift_index] = score
            row_finite[row, shift_index] = finite
            row_sites[row, shift_index] = sites

    for shift_index in range(n_shifts):
        total_row_score = 0.0
        total_finite = 0
        total_sites = 0
        for row in range(n_rows):
            total_row_score += row_scores[row, shift_index]
            total_finite += row_finite[row, shift_index]
            total_sites += row_sites[row, shift_index]
        out_scores[shift_index] = (
            0.0
            if total_sites == 0 or total_finite == 0
            else total_row_score / total_finite
        )
        out_sites[shift_index] = total_sites


@njit(cache=True)
def _score_shift_best(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    query_site_start_offset, target_site_start_offset,
    shift, window_radius, realign_window,
    metric_kind, use_dice,
    out_score, out_sites,
):
    n_rows = scores1_offsets.shape[0] - 1
    total_row_score = 0.0
    total_finite = 0
    total_sites = 0

    for row in range(n_rows):
        row_score, row_finite, row_sites = _score_row_best(
            scores1_data, scores1_offsets,
            scores2_data, scores2_offsets,
            query_positions, query_offsets,
            target_positions, target_offsets,
            query_site_start_offset, target_site_start_offset,
            row, shift, window_radius, realign_window,
            metric_kind, use_dice,
        )
        total_row_score += row_score
        total_finite += row_finite
        total_sites += row_sites
    out_score[0] = 0.0 if total_sites == 0 or total_finite == 0 else total_row_score / total_finite
    out_sites[0] = total_sites


@njit(parallel=True, cache=True)
def _score_orientation_best_parallel(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    query_site_start_offset, target_site_start_offset,
    search_range, window_radius, realign_window,
    metric_kind, use_dice,
    row_scores, row_finite, row_sites, out_scores, out_sites,
):
    n_rows = scores1_offsets.shape[0] - 1
    n_shifts = 2 * search_range + 1
    for row in prange(n_rows):
        for shift_index in range(n_shifts):
            shift = shift_index - search_range
            score, finite, sites = _score_row_best(
                scores1_data, scores1_offsets,
                scores2_data, scores2_offsets,
                query_positions, query_offsets,
                target_positions, target_offsets,
                query_site_start_offset, target_site_start_offset,
                row, shift, window_radius, realign_window,
                metric_kind, use_dice,
            )
            row_scores[row, shift_index] = score
            row_finite[row, shift_index] = finite
            row_sites[row, shift_index] = sites

    for shift_index in range(n_shifts):
        total_row_score = 0.0
        total_finite = 0
        total_sites = 0
        for row in range(n_rows):
            total_row_score += row_scores[row, shift_index]
            total_finite += row_finite[row, shift_index]
            total_sites += row_sites[row, shift_index]
        out_scores[shift_index] = (
            0.0
            if total_sites == 0 or total_finite == 0
            else total_row_score / total_finite
        )
        out_sites[shift_index] = total_sites
