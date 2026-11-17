"""Empirical log-tail normalization strategies."""

from __future__ import annotations

import numpy as np

from ..arrays import RaggedArray, StrandPair
from ..parallel import use_parallel
from .._kernels import (
    count_unique_sorted,
    fill_empirical_table_sorted,
    transform_empirical_scores,
    transform_empirical_scores_parallel,
)


class LogTailTable:
    __slots__ = ("scores", "log_tail")

    def __init__(self, scores, log_tail):
        self.scores = np.ascontiguousarray(scores, dtype=np.float32)
        self.log_tail = np.ascontiguousarray(log_tail, dtype=np.float32)


class EmpiricalLogTail:
    """Exact empirical upper-tail normalization."""

    def __eq__(self, other):
        return isinstance(other, EmpiricalLogTail)

    def __hash__(self):
        return hash("EmpiricalLogTail")


def normalization_fingerprint(strategy):
    if isinstance(strategy, EmpiricalLogTail):
        return "empirical-log-tail-v2"
    raise ValueError(f"unknown normalization strategy: {strategy!r}")


def _fit_empirical_table(values):
    n = values.size
    if n == 0:
        return LogTailTable(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    unique, counts = np.unique(values, return_counts=True)
    unique = unique[::-1]
    counts = counts[::-1]
    cumulative = np.cumsum(counts, dtype=np.float64)
    log_tail = (-np.log10(cumulative / n)).astype(np.float32)
    return LogTailTable(unique.astype(np.float32), log_tail)


def fit(strategy, scores):
    values = np.asarray(scores, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("normalization scores must be finite.")
    normalization_fingerprint(strategy)
    return _fit_empirical_table(values)


def transform_scores(table, scores, *, out=None):
    if out is None:
        out = np.empty(scores.data.size, dtype=np.float32)
    elif out is not scores.data:
        raise ValueError("normalization output must be the input score buffer.")
    parallel = use_parallel(scores.data.size)
    if not isinstance(table, LogTailTable):
        raise ValueError(f"unknown table type: {type(table)!r}")
    kernel = transform_empirical_scores_parallel if parallel else transform_empirical_scores
    kernel(scores.data, table.scores, table.log_tail, out)
    return RaggedArray(out, scores.offsets)


def normalize_bundle(table, bundle, *, in_place=False):
    fwd = transform_scores(
        table,
        bundle.forward,
        out=bundle.forward.data if in_place else None,
    )
    if bundle.forward is bundle.reverse:
        return StrandPair(fwd, fwd)
    return StrandPair(
        fwd,
        transform_scores(
            table,
            bundle.reverse,
            out=bundle.reverse.data if in_place else None,
        ),
    )


def _fit_exact(strategy, bundle):
    """Build one threshold-independent exact empirical table."""
    normalization_fingerprint(strategy)
    fwd = bundle.forward.data
    rev = bundle.reverse.data
    if bundle.forward is bundle.reverse:
        values = fwd.copy()
    else:
        values = np.empty(fwd.size + rev.size, dtype=np.float32)
        values[: fwd.size] = fwd
        values[fwd.size :] = rev
    if values.size == 0:
        return LogTailTable(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )
    values.sort()
    n_unique = count_unique_sorted(values)
    scores = np.empty(n_unique, dtype=np.float32)
    log_tail = np.empty(n_unique, dtype=np.float32)
    fill_empirical_table_sorted(values, values.size, scores, log_tail)
    return LogTailTable(scores, log_tail)
