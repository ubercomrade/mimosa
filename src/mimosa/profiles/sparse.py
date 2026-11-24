"""Bounded, optional sparse Jaccard kernels for Monte Carlo.

Anchors select the union where either score reaches the threshold. Values
below threshold on the other profile still contribute to weighted Jaccard.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..backend import _compare_circular


@jax.jit
def _anchor_counts(scores, threshold):
    return jnp.sum(scores >= threshold, axis=-1, dtype=jnp.int32)


@partial(jax.jit, static_argnames=("capacity",))
def _extract(scores, counts, threshold, *, capacity):
    shape = scores.shape[:-1]
    rows = scores.reshape((-1, scores.shape[-1]))
    indices = jax.vmap(lambda row: jnp.nonzero(
        row >= threshold, size=capacity, fill_value=0,
    )[0].astype(jnp.int32))(rows)
    valid = jnp.arange(capacity)[None, :] < counts.reshape((-1, 1))
    return indices.reshape((*shape, capacity)), valid.reshape((*shape, capacity))


def make_anchors(scores, threshold):
    """Return device anchors or None when a dense comparison is preferable."""
    n = scores.shape[-1]
    if n == 0:
        return None
    counts = _anchor_counts(scores, threshold)
    max_count = int(np.max(np.asarray(counts), initial=0))
    # Avoid enormous padded arrays and pointless sparse work for dense tracks.
    if max_count > max(1, n // 8):
        return None
    capacity = max(1, 1 << max(0, max_count - 1).bit_length())
    return _extract(scores, counts, threshold, capacity=capacity)


@partial(jax.jit, static_argnames=("search_range",))
def _sparse_score(query, target, qidx, qvalid, tidx, tvalid, starts, sizes,
                  threshold, *, search_range):
    shifts = 2 * search_range + 1

    def one(index, best):
        orientation = index // shifts
        delta = index % shifts - search_range
        qs, ts = orientation // 2, orientation % 2
        qp, tp = qidx[qs], tidx[ts]
        qpartner = starts[qp] + (qp - starts[qp] + delta) % sizes[qp]
        tqpos = starts[tp] + (tp - starts[tp] - delta) % sizes[tp]
        qa, qb = query[qs, qp], target[ts, qpartner]
        ta, tb = query[qs, tqpos], target[ts, tp]
        tmask = tvalid[ts] & (ta < threshold)
        low = (jnp.sum(jnp.where(qvalid[qs], jnp.minimum(qa, qb), 0).astype(jnp.float64))
               + jnp.sum(jnp.where(tmask, jnp.minimum(ta, tb), 0).astype(jnp.float64)))
        high = (jnp.sum(jnp.where(qvalid[qs], jnp.maximum(qa, qb), 0).astype(jnp.float64))
                + jnp.sum(jnp.where(tmask, jnp.maximum(ta, tb), 0).astype(jnp.float64)))
        return jnp.maximum(best, jnp.where(high > 0, low / high, 0))

    return jax.lax.fori_loop(0, 4 * shifts, one, jnp.float64(0)).astype(jnp.float32)


@partial(jax.jit, static_argnames=("search_range", "return_mask"))
def count_exceedances(queries, target, qidx, qvalid, tidx, tvalid, starts, sizes,
                      threshold, observed, *, search_range, return_mask=False):
    """Check sparse exceedances, resolving near-ties with the dense kernel."""
    sparse = jax.vmap(lambda q, qi, qv: _sparse_score(
        q, target, qi, qv, tidx, tvalid, starts, sizes, threshold,
        search_range=search_range,
    ))(queries, qidx, qvalid)
    # The different reduction order can round near an observed score. Recheck
    # the entire block on the dense path around the float32 decision boundary.
    margin = jnp.float32(16 * np.finfo(np.float32).eps)
    near = jnp.any(jnp.abs(sparse - observed) <= margin)

    def dense(_):
        return jax.vmap(lambda q: _compare_circular(
            q, target, starts, sizes, threshold, search_range=search_range,
        )[0])(queries)

    scores = jax.lax.cond(near, dense, lambda _: sparse, operand=None)
    hits = scores >= observed
    return hits if return_mask else jnp.sum(hits, dtype=jnp.int64)
