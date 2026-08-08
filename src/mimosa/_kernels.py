"""Numba kernels: scanning, best-strand reduction, anchor collection, alignment.

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
    codes = np.empty(n_codes, dtype=np.int64)
    ho_kmer_codes(seq, kmer_size, 0, n_codes, False, codes)
    for pos in range(n_pos):
        total = np.float32(0.0)
        for term in range(n_terms):
            total += weights[codes[pos + term], term]
        out[pos] = total


@njit(cache=True)
def rolling_scan_reverse(weights, seq, kmer_size, n_terms, n_pos, out):
    n_codes = n_pos + n_terms - 1
    codes = np.empty(n_codes, dtype=np.int64)
    ho_kmer_codes(seq, kmer_size, 0, n_codes, True, codes)
    for pos in range(n_pos):
        total = np.float32(0.0)
        for term in range(n_terms):
            total += weights[codes[n_pos - pos - 1 + term], term]
        out[pos] = total


@njit(cache=True)
def best_strand_reduce(forward, reverse, out):
    n = forward.shape[0]
    for i in range(n):
        out[i] = reverse[i] if reverse[i] > forward[i] else forward[i]


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


@njit(parallel=True, cache=True)
def batch_pwm_models_forward(
    weights, lengths, model_indices, seq_data, seq_offsets, out_data, out_offsets
):
    for local_model in prange(model_indices.shape[0]):
        model = model_indices[local_model]
        n_terms = lengths[local_model]
        for row in range(seq_offsets.shape[0] - 1):
            sequence_start = seq_offsets[row]
            n_pos = out_offsets[model, row + 1] - out_offsets[model, row]
            if n_pos <= 0:
                continue
            pwm_scan_forward(
                weights[local_model],
                seq_data[sequence_start : seq_offsets[row + 1]],
                n_pos,
                n_terms,
                out_data[out_offsets[model, row] : out_offsets[model, row + 1]],
            )


@njit(parallel=True, cache=True)
def batch_pwm_models_reverse(
    weights, lengths, model_indices, seq_data, seq_offsets, out_data, out_offsets
):
    for local_model in prange(model_indices.shape[0]):
        model = model_indices[local_model]
        n_terms = lengths[local_model]
        for row in range(seq_offsets.shape[0] - 1):
            sequence_start = seq_offsets[row]
            n_pos = out_offsets[model, row + 1] - out_offsets[model, row]
            if n_pos <= 0:
                continue
            pwm_scan_reverse(
                weights[local_model],
                seq_data[sequence_start : seq_offsets[row + 1]],
                n_pos,
                n_terms,
                out_data[out_offsets[model, row] : out_offsets[model, row + 1]],
            )


@njit(parallel=True, cache=True)
def batch_rolling_models_forward(
    weights,
    model_indices,
    seq_data,
    seq_offsets,
    out_data,
    out_offsets,
    kmer_size,
    n_terms,
):
    for local_model in prange(model_indices.shape[0]):
        model = model_indices[local_model]
        for row in range(seq_offsets.shape[0] - 1):
            sequence_start = seq_offsets[row]
            n_pos = out_offsets[model, row + 1] - out_offsets[model, row]
            if n_pos <= 0:
                continue
            rolling_scan_forward(
                weights[local_model],
                seq_data[sequence_start : seq_offsets[row + 1]],
                kmer_size,
                n_terms,
                n_pos,
                out_data[out_offsets[model, row] : out_offsets[model, row + 1]],
            )


@njit(parallel=True, cache=True)
def batch_rolling_models_reverse(
    weights,
    model_indices,
    seq_data,
    seq_offsets,
    out_data,
    out_offsets,
    kmer_size,
    n_terms,
):
    for local_model in prange(model_indices.shape[0]):
        model = model_indices[local_model]
        for row in range(seq_offsets.shape[0] - 1):
            sequence_start = seq_offsets[row]
            n_pos = out_offsets[model, row + 1] - out_offsets[model, row]
            if n_pos <= 0:
                continue
            rolling_scan_reverse(
                weights[local_model],
                seq_data[sequence_start : seq_offsets[row + 1]],
                kmer_size,
                n_terms,
                n_pos,
                out_data[out_offsets[model, row] : out_offsets[model, row + 1]],
            )


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
def _lower_bound_desc(scores, target):
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
        if scores[mid] > target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def transform_empirical_scores(scores, table_scores, table_log_tail, out):
    for i in range(scores.shape[0]):
        out[i] = table_log_tail[_lower_bound_desc(table_scores, scores[i])]


@njit(parallel=True, cache=True)
def transform_empirical_scores_parallel(scores, table_scores, table_log_tail, out):
    for i in prange(scores.shape[0]):
        out[i] = table_log_tail[_lower_bound_desc(table_scores, scores[i])]


@njit(cache=True)
def _transform_hybrid_one(
    score, minimum, bin_width, histogram_log_tail, exact_scores, exact_log_tail
):
    n_bins = histogram_log_tail.shape[0]
    exact_size = exact_scores.shape[0]
    if exact_size > 0 and score >= exact_scores[exact_size - 1]:
        return exact_log_tail[_lower_bound_desc(exact_scores, score)]
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
def _score_shift_csr(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    shift, window_radius, realign_window,
    metric_kind,  # 0=rowwise, 1=cosine
    use_dice, seen, candidates,
    out_score, out_sites,
):
    n_rows = scores1_offsets.shape[0] - 1
    total_row_score = 0.0
    total_finite = 0
    total_sites = 0

    seen.fill(0)

    for row in range(n_rows):
        len1 = scores1_offsets[row + 1] - scores1_offsets[row]
        len2 = scores2_offsets[row + 1] - scores2_offsets[row]
        r1_start = scores1_offsets[row]
        r2_start = scores2_offsets[row]

        # collect candidates with epoch dedup
        epoch = row + 1
        count = 0

        for idx in range(query_offsets[row], query_offsets[row + 1]):
            pos1 = query_positions[idx]
            pos2 = pos1 + shift
            if _window_fits(pos1, len1, window_radius) and _window_fits(pos2, len2, window_radius):
                if seen[pos1] != epoch:
                    seen[pos1] = epoch
                    candidates[count] = pos1
                    count += 1

        for idx in range(target_offsets[row], target_offsets[row + 1]):
            expected_pos1 = target_positions[idx] - shift
            pos1 = _realign_query_position(scores1_data[r1_start:r1_start + len1], expected_pos1, realign_window)
            if pos1 < 0:
                continue
            pos2 = pos1 + shift
            if _window_fits(pos1, len1, window_radius) and _window_fits(pos2, len2, window_radius):
                if seen[pos1] != epoch:
                    seen[pos1] = epoch
                    candidates[count] = pos1
                    count += 1

        total_sites += count
        if count == 0:
            continue

        for c in range(count):
            pos1 = candidates[c]
            if metric_kind == 1:
                s_sum, f_count = _accumulate_cosine(
                    scores1_data[r1_start:r1_start + len1], scores2_data[r2_start:r2_start + len2], pos1, shift, window_radius
                )
                total_row_score += s_sum
                total_finite += f_count
            else:
                s_sum, f_count = _accumulate_overlap(
                    scores1_data[r1_start:r1_start + len1], scores2_data[r2_start:r2_start + len2], pos1, shift, window_radius, use_dice
                )
                total_row_score += s_sum
                total_finite += f_count

    if total_sites == 0:
        out_score[0] = 0.0
        out_sites[0] = 0
        return

    score = 0.0 if total_finite == 0 else total_row_score / total_finite

    out_score[0] = score
    out_sites[0] = total_sites


@njit(cache=True)
def _score_shift_best(
    scores1_data, scores1_offsets,
    scores2_data, scores2_offsets,
    query_positions, query_offsets,
    target_positions, target_offsets,
    shift, window_radius, realign_window,
    metric_kind, use_dice,
    out_score, out_sites,
):
    n_rows = scores1_offsets.shape[0] - 1
    total_row_score = 0.0
    total_finite = 0
    total_sites = 0

    for row in range(n_rows):
        len1 = scores1_offsets[row + 1] - scores1_offsets[row]
        len2 = scores2_offsets[row + 1] - scores2_offsets[row]
        r1_start = scores1_offsets[row]
        r2_start = scores2_offsets[row]
        r1 = scores1_data[r1_start:r1_start + len1]
        r2 = scores2_data[r2_start:r2_start + len2]
        query_pos = -1
        target_pos = -1

        if query_offsets[row] < query_offsets[row + 1]:
            candidate = query_positions[query_offsets[row]]
            if _window_fits(candidate, len1, window_radius) and _window_fits(candidate + shift, len2, window_radius):
                query_pos = candidate

        if target_offsets[row] < target_offsets[row + 1]:
            expected = target_positions[target_offsets[row]] - shift
            candidate = _realign_query_position(r1, expected, realign_window)
            if candidate >= 0 and _window_fits(candidate, len1, window_radius) and _window_fits(candidate + shift, len2, window_radius):
                target_pos = candidate

        if query_pos >= 0:
            total_sites += 1
            if metric_kind == 1:
                s_sum, f_count = _accumulate_cosine(r1, r2, query_pos, shift, window_radius)
                total_row_score += s_sum
                total_finite += f_count
            else:
                s_sum, f_count = _accumulate_overlap(r1, r2, query_pos, shift, window_radius, use_dice)
                total_row_score += s_sum
                total_finite += f_count

        if target_pos >= 0 and target_pos != query_pos:
            total_sites += 1
            if metric_kind == 1:
                s_sum, f_count = _accumulate_cosine(r1, r2, target_pos, shift, window_radius)
                total_row_score += s_sum
                total_finite += f_count
            else:
                s_sum, f_count = _accumulate_overlap(r1, r2, target_pos, shift, window_radius, use_dice)
                total_row_score += s_sum
                total_finite += f_count

    if total_sites == 0:
        out_score[0] = 0.0
        out_sites[0] = 0
        return

    score = 0.0 if total_finite == 0 else total_row_score / total_finite

    out_score[0] = score
    out_sites[0] = total_sites


@njit(cache=True)
def _batch_orientation_best(
    query_scores_data,
    query_scores_offsets,
    target_scores_data,
    target_scores_offsets,
    query_positions,
    query_anchor_offsets,
    target_positions,
    target_anchor_offsets,
    shift_range,
    window_radius,
    realign_window,
    metric_kind,
    use_dice,
    min_logerr,
    seen,
    candidates,
    score_work,
    sites_work,
    target_index,
):
    best_score = np.float32(0.0)
    best_shift = 0
    best_sites = 0
    out_score = score_work[target_index : target_index + 1]
    out_sites = sites_work[target_index : target_index + 1]

    for shift_index in range(2 * shift_range + 1):
        shift = shift_index - shift_range
        if min_logerr > 0.0:
            _score_shift_csr(
                query_scores_data,
                query_scores_offsets,
                target_scores_data,
                target_scores_offsets,
                query_positions,
                query_anchor_offsets,
                target_positions,
                target_anchor_offsets,
                shift,
                window_radius,
                realign_window,
                metric_kind,
                use_dice,
                seen[target_index],
                candidates[target_index],
                out_score,
                out_sites,
            )
        else:
            _score_shift_best(
                query_scores_data,
                query_scores_offsets,
                target_scores_data,
                target_scores_offsets,
                query_positions,
                query_anchor_offsets,
                target_positions,
                target_anchor_offsets,
                shift,
                window_radius,
                realign_window,
                metric_kind,
                use_dice,
                out_score,
                out_sites,
            )
        score = np.float32(out_score[0])
        n_sites = out_sites[0]
        if float(score) > float(best_score) or (
            float(score) == float(best_score)
            and (
                n_sites > best_sites
                or (n_sites == best_sites and abs(shift) < abs(best_shift))
            )
        ):
            best_score = score
            best_shift = shift
            best_sites = n_sites
    return best_score, best_shift, best_sites


@njit(parallel=True, cache=True)
def batch_profile_compare(
    query_forward_data,
    query_forward_offsets,
    query_reverse_data,
    query_reverse_offsets,
    query_forward_positions,
    query_forward_anchor_offsets,
    query_reverse_positions,
    query_reverse_anchor_offsets,
    target_forward_data,
    target_forward_offsets,
    target_reverse_data,
    target_reverse_offsets,
    target_forward_positions,
    target_forward_anchor_offsets,
    target_reverse_positions,
    target_reverse_anchor_offsets,
    target_shared,
    query_shared,
    search_range,
    window_radius,
    realign_window,
    metric_kind,
    use_dice,
    min_logerr,
    seen,
    candidates,
    score_work,
    sites_work,
    out_scores,
    out_shifts,
    out_orientations,
    out_sites,
):
    n_targets = target_shared.shape[0]
    for target_index in prange(n_targets):
        best_score = np.float32(0.0)
        best_shift = 0
        best_orientation = 0
        best_sites = 0
        best_rank = 2**63 - 1
        n_query_strands = 1 if query_shared else 2
        n_target_strands = 1 if target_shared[target_index] else 2

        for query_strand in range(n_query_strands):
            if query_strand == 0:
                query_data = query_forward_data
                query_offsets = query_forward_offsets
                query_positions = query_forward_positions
                query_anchor_offsets = query_forward_anchor_offsets
            else:
                query_data = query_reverse_data
                query_offsets = query_reverse_offsets
                query_positions = query_reverse_positions
                query_anchor_offsets = query_reverse_anchor_offsets

            for target_strand in range(n_target_strands):
                orientation = query_strand * 2 + target_strand
                if target_strand == 0:
                    target_data = target_forward_data
                    target_offsets = target_forward_offsets[target_index]
                    target_positions = target_forward_positions
                    target_anchor_offsets = target_forward_anchor_offsets[target_index]
                else:
                    target_data = target_reverse_data
                    target_offsets = target_reverse_offsets[target_index]
                    target_positions = target_reverse_positions
                    target_anchor_offsets = target_reverse_anchor_offsets[target_index]

                score, shift, n_sites = _batch_orientation_best(
                    query_data,
                    query_offsets,
                    target_data,
                    target_offsets,
                    query_positions,
                    query_anchor_offsets,
                    target_positions,
                    target_anchor_offsets,
                    search_range,
                    window_radius,
                    realign_window,
                    metric_kind,
                    use_dice,
                    min_logerr,
                    seen,
                    candidates,
                    score_work,
                    sites_work,
                    target_index,
                )
                rank = orientation
                if float(score) > float(best_score) or (
                    float(score) == float(best_score)
                    and (
                        n_sites > best_sites
                        or (
                            n_sites == best_sites
                            and (
                                abs(shift) < abs(best_shift)
                                or (
                                    abs(shift) == abs(best_shift)
                                    and rank < best_rank
                                )
                            )
                        )
                    )
                ):
                    best_score = score
                    best_shift = shift
                    best_orientation = orientation
                    best_sites = n_sites
                    best_rank = rank

        out_scores[target_index] = best_score
        out_shifts[target_index] = best_shift
        out_orientations[target_index] = best_orientation
        out_sites[target_index] = best_sites
