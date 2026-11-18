"""Backend primitives for JAX scanning, normalization, and comparison.

Scanning and normalization are used by the public profile pipeline. Arrays
stay on the selected JAX device until the caller converts them to NumPy.
Normalization and comparison require a local or global JAX x64 setting;
this module does not change the process-wide JAX configuration.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .arrays import EncodedSequences
from .models import BaMM, Dimont, PWM, Slim, strict_integer
from .profiles.anchors import parse_min_logerr


def _require_x64():
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX x64 is required; set JAX_ENABLE_X64=1 before starting Python.")


@partial(jax.jit, static_argnames=("width", "order"))
def _circular_scan(data, row_start, row_length, weights, *, width, order):
    position = jnp.arange(data.size, dtype=jnp.int32) - row_start

    def base_term(term, scores):
        def code_term(j, codes):
            a, b = codes
            fpos = (position - order + term + j) % row_length
            rpos = (position + width + order - 1 - term - j) % row_length
            f = data[row_start + fpos].astype(jnp.int32)
            r = data[row_start + rpos].astype(jnp.int32)
            r = jnp.where(r == 4, 4, 3 - r)
            return 5 * a + f, 5 * b + r

        zero = jnp.zeros(data.shape, dtype=jnp.int32)
        fcode, rcode = jax.lax.fori_loop(0, order + 1, code_term, (zero, zero))
        return scores[0] + weights[fcode, term], scores[1] + weights[rcode, term]

    zero = jnp.zeros(data.shape, dtype=jnp.float32)
    return jax.lax.fori_loop(0, width, base_term, (zero, zero))


@partial(jax.jit, static_argnames=("width", "order"))
def _window_site_scores(weights, windows, *, width, order):
    """Score forward-oriented windows whose sites start at the window origin.

    ``width`` is the motif length. Code accumulation and Float32 summation
    follow the sequence-level kernels exactly, so both paths agree bitwise.
    """
    total = jnp.zeros(windows.shape[0], dtype=jnp.float32)
    for term in range(width):
        code = jnp.zeros(windows.shape[0], dtype=jnp.int32)
        for j in range(order + 1):
            code = 5 * code + windows[:, term + j].astype(jnp.int32)
        total = total + weights[code, term]
    return total


def window_site_scores(model, windows):
    """Score oriented windows with a built-in model's table."""
    if not isinstance(model, (PWM, BaMM, Dimont, Slim)):
        raise TypeError("window site scoring supports PWM, BaMM, Dimont and Slim only.")
    return _window_site_scores(
        jnp.asarray(model.weights), jnp.asarray(windows),
        width=model.motif_length,
        order=0 if isinstance(model, PWM) else model.order,
    )


def scan_circular(model, sequences):
    """Scan both strands at each position of ragged circular sequences.

    Return two device arrays in the same flat order as ``sequences.data``.
    Supports the four built-in families; SiteGA and custom models are not
    silently routed through this implementation.
    """
    if not isinstance(model, (PWM, BaMM, Dimont, Slim)):
        raise TypeError("JAX circular scan supports PWM, BaMM, Dimont and Slim only.")
    if not isinstance(sequences, EncodedSequences):
        raise TypeError("sequences must be an EncodedSequences batch.")
    if sequences.data.size > np.iinfo(np.int32).max:
        raise ValueError("JAX scan currently supports at most 2**31-1 positions.")
    lengths = np.diff(sequences.offsets)
    starts = np.repeat(sequences.offsets[:-1], lengths).astype(np.int32)
    sizes = np.repeat(lengths, lengths).astype(np.int32)
    return _circular_scan(
        jnp.asarray(sequences.data), jnp.asarray(starts), jnp.asarray(sizes),
        jnp.asarray(model.weights), width=model.motif_length,
        order=0 if isinstance(model, PWM) else model.order,
    )


@partial(jax.jit, static_argnames=("width", "order"))
def _linear_scan(data, row_start, output_start, weights, *, width, order):
    position = jnp.arange(row_start.size, dtype=jnp.int32) - output_start

    def base_term(term, scores):
        def code_term(j, codes):
            a, b = codes
            f = data[row_start + position + term + j].astype(jnp.int32)
            r = data[row_start + position + width + 2 * order - 1 - term - j].astype(jnp.int32)
            return 5 * a + f, 5 * b + jnp.where(r == 4, 4, 3 - r)

        zero = jnp.zeros(row_start.shape, dtype=jnp.int32)
        fcode, rcode = jax.lax.fori_loop(0, order + 1, code_term, (zero, zero))
        return scores[0] + weights[fcode, term], scores[1] + weights[rcode, term]

    zero = jnp.zeros(row_start.shape, dtype=jnp.float32)
    return jax.lax.fori_loop(0, width, base_term, (zero, zero))


def scan_linear(model, sequences):
    """Return two device score arrays and host offsets for linear sequences."""
    if not isinstance(model, (PWM, BaMM, Dimont, Slim)):
        raise TypeError("JAX linear scan supports PWM, BaMM, Dimont and Slim only.")
    if not isinstance(sequences, EncodedSequences):
        raise TypeError("sequences must be an EncodedSequences batch.")
    if sequences.data.size > np.iinfo(np.int32).max:
        raise ValueError("JAX scan currently supports at most 2**31-1 positions.")
    order = 0 if isinstance(model, PWM) else model.order
    lengths = np.diff(sequences.offsets)
    counts = np.maximum(lengths - (model.motif_length + 2 * order) + 1, 0)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    if not offsets[-1]:
        empty = jnp.empty(0, dtype=jnp.float32)
        return (empty, empty), offsets
    starts = np.repeat(sequences.offsets[:-1], counts).astype(np.int32)
    out_starts = np.repeat(offsets[:-1], counts).astype(np.int32)
    result = _linear_scan(
        jnp.asarray(sequences.data), jnp.asarray(starts), jnp.asarray(out_starts),
        jnp.asarray(model.weights), width=model.motif_length, order=order,
    )
    return result, offsets


@jax.jit
def _normalize_exact(scores, calibration):
    if calibration.size == 0:
        return jnp.zeros_like(scores)
    ordered = jnp.sort(calibration)
    indices = jnp.searchsorted(ordered, scores, side="left", method="scan")
    # CPU table clamps to the maximal calibration score when scores exceed it.
    # Looking up its first occurrence also preserves ties at that boundary.
    selected = ordered[jnp.minimum(indices, calibration.size - 1)]
    first = jnp.searchsorted(ordered, selected, side="left", method="scan")
    tail = (calibration.size - first).astype(jnp.float64) / jnp.float64(calibration.size)
    # XLA may replace division by a reciprocal, so n/n can round below 1.
    # The exact empirical survival probability at the minimum is one.
    return jnp.where(first == 0, jnp.float32(-0.0),
                     (-jnp.log10(tail)).astype(jnp.float32))


def normalize_exact(scores, calibration=None):
    """Apply the exact empirical table to flat Float32 scores on device."""
    _require_x64()
    scores = jnp.asarray(scores, dtype=jnp.float32)
    calibration = scores if calibration is None else jnp.asarray(calibration, dtype=jnp.float32)
    if scores.ndim != 1 or calibration.ndim != 1:
        raise ValueError("scores and calibration must be one-dimensional.")
    return _normalize_exact(scores, calibration)


@partial(jax.jit, static_argnames=("search_range",))
def _compare_circular(query, target, row_start, row_length, threshold, *, search_range):
    position = jnp.arange(row_start.size, dtype=jnp.int32) - row_start
    n_shifts = 2 * search_range + 1

    def score_one(index, best):
        score, shift, orientation, sites = best
        orient = index // n_shifts
        offset = index % n_shifts - search_range
        q = query[orient // 2]
        partner = row_start + (position + offset) % row_length
        t = target[orient % 2, partner]
        mask = (q >= threshold) | (t >= threshold)
        low = jnp.where(mask, jnp.minimum(q, t), 0).astype(jnp.float64)
        high = jnp.where(mask, jnp.maximum(q, t), 0).astype(jnp.float64)
        numerator = jnp.sum(low)
        denominator = jnp.sum(high)
        current = jnp.where(denominator > 0, numerator / denominator, 0)
        count = jnp.sum(mask, dtype=jnp.int64)
        take = (current > score) | (
            (current == score)
            & ((count > sites) | ((count == sites) & (jnp.abs(offset) < jnp.abs(shift))))
        )
        return (jnp.where(take, current, score), jnp.where(take, offset, shift),
                jnp.where(take, orient, orientation), jnp.where(take, count, sites))

    initial = (jnp.float64(0), jnp.int32(0), jnp.int32(0), jnp.int64(0))
    result = jax.lax.fori_loop(0, 4 * n_shifts, score_one, initial)
    return result[0].astype(jnp.float32), result[1], result[2], result[3]


def compare_circular(query, target, offsets, *, min_logerr, search_range=5):
    """Return device scalar score, shift, orientation index, and site count.

    Dense Float64 reductions use inputs of shape ``(2, positions)``.
    """
    _require_x64()
    search_range = strict_integer(search_range, "search_range")
    if search_range < 0:
        raise ValueError("search_range must be non-negative.")
    threshold = parse_min_logerr(min_logerr)
    query, target = jnp.asarray(query, dtype=jnp.float32), jnp.asarray(target, dtype=jnp.float32)
    if query.ndim != 2 or query.shape[0] != 2 or query.shape != target.shape:
        raise ValueError("query and target must have the same shape (2, positions).")
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or offsets.size == 0 or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != query.shape[1]
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("offsets must describe the flattened profile rows.")
    if query.shape[1] > np.iinfo(np.int32).max:
        raise ValueError("JAX comparison currently supports at most 2**31-1 positions.")
    lengths = np.diff(offsets)
    starts = np.repeat(offsets[:-1], lengths).astype(np.int32)
    sizes = np.repeat(lengths, lengths).astype(np.int32)
    if query.shape[1] == 0:
        return (jnp.float32(0), jnp.int32(0), jnp.int32(0), jnp.int64(0))
    return _compare_circular(query, target, jnp.asarray(starts), jnp.asarray(sizes),
                             threshold, search_range=search_range)
