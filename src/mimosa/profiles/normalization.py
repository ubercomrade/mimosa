"""Empirical log-tail normalization strategies."""

from __future__ import annotations

import math

import numpy as np

from ..arrays import RaggedArray, StrandPair
from ..parallel import use_parallel
from .._kernels import (
    transform_empirical_scores,
    transform_empirical_scores_parallel,
    transform_hybrid_scores,
    transform_hybrid_scores_parallel,
)


class LogTailTable:
    __slots__ = ("scores", "log_tail")

    def __init__(self, scores, log_tail):
        self.scores = np.ascontiguousarray(scores, dtype=np.float32)
        self.log_tail = np.ascontiguousarray(log_tail, dtype=np.float32)


class HybridLogTailTable:
    __slots__ = ("minimum", "bin_width", "log_tail", "exact_tail")

    def __init__(self, minimum, bin_width, log_tail, exact_tail):
        self.minimum = np.float32(minimum)
        self.bin_width = float(bin_width)
        self.log_tail = np.ascontiguousarray(log_tail, dtype=np.float32)
        self.exact_tail = exact_tail


class EmpiricalLogTail:
    def __eq__(self, other):
        return isinstance(other, EmpiricalLogTail)

    def __hash__(self):
        return hash("EmpiricalLogTail")


class HybridEmpiricalLogTail:
    def __init__(self, bins=65_536):
        if not (256 <= bins <= 1_048_576):
            raise ValueError("hybrid normalization bins must be in 256:1_048_576.")
        if bins & (bins - 1) != 0:
            raise ValueError("hybrid normalization bins must be a power of two.")
        self.bins = int(bins)

    def __eq__(self, other):
        return isinstance(other, HybridEmpiricalLogTail) and self.bins == other.bins

    def __hash__(self):
        return hash(("HybridEmpiricalLogTail", self.bins))


def normalization_fingerprint(strategy):
    if isinstance(strategy, EmpiricalLogTail):
        return "empirical-log-tail-v1"
    if isinstance(strategy, HybridEmpiricalLogTail):
        return f"hybrid-log-tail-v2;bins={strategy.bins}"
    raise ValueError(f"unknown normalization strategy: {strategy!r}")


def _fit_empirical_table(values, total_n=None):
    n = values.size
    if total_n is None:
        total_n = n
    if n == 0:
        return LogTailTable(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    sorted_desc = np.sort(values)[::-1]
    unique, counts = np.unique(sorted_desc, return_counts=True)
    unique = unique[::-1]
    counts = counts[::-1]
    cumulative = np.cumsum(counts, dtype=np.float64)
    log_tail = (-np.log10(cumulative / total_n)).astype(np.float32)
    return LogTailTable(unique.astype(np.float32), log_tail)


def fit(strategy, scores, *, tail_logerr=0.0):
    values = np.asarray(scores, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("normalization scores must be finite.")
    if isinstance(strategy, EmpiricalLogTail):
        return _fit_empirical_table(values)
    if not isinstance(strategy, HybridEmpiricalLogTail):
        raise ValueError(f"unknown normalization strategy: {strategy!r}")
    n = values.size
    if n == 0:
        return HybridLogTailTable(
            0.0, 1.0, np.array([0.0], dtype=np.float32),
            LogTailTable(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)),
        )
    lo = float(values.min())
    hi = float(values.max())
    width = 1.0 if lo == hi else (hi - lo) / strategy.bins
    bins = 1 if lo == hi else strategy.bins
    if lo == hi:
        bin_indices = np.ones(n, dtype=np.int64)
    else:
        bin_indices = np.floor((values.astype(np.float64) - lo) / width).astype(np.int64) + 1
        bin_indices = np.clip(bin_indices, 1, bins)
    counts = np.bincount(bin_indices, minlength=bins + 1)[1 : bins + 1].astype(np.uint64)
    cumulative = np.cumsum(counts[::-1], dtype=np.uint64)[::-1]
    if not math.isfinite(tail_logerr):
        raise ValueError("tail_logerr must be finite.")
    effective = max(0.0, float(tail_logerr))
    cutoff_count = max(1, math.ceil(n * 10.0 ** (-effective)))
    ge = np.flatnonzero(cumulative >= cutoff_count)
    cutoff_bin = int(ge[-1]) + 1 if ge.size else 1
    tail_mask = bin_indices >= cutoff_bin
    exact = _fit_empirical_table(values[tail_mask], total_n=n)
    histogram_log_tail = (-np.log10(cumulative.astype(np.float64) / n)).astype(np.float32)
    return HybridLogTailTable(lo, width, histogram_log_tail, exact)


def lookup_score(table, score):
    if isinstance(table, LogTailTable):
        idx = min(
            int(np.searchsorted(-table.scores, -score, side="left")),
            table.scores.size - 1,
        )
        return table.log_tail[idx]
    if isinstance(table, HybridLogTailTable):
        if table.exact_tail.scores.size and score >= table.exact_tail.scores[-1]:
            return lookup_score(table.exact_tail, score)
        if table.log_tail.size == 0:
            return np.float32(0.0)
        if table.bin_width == 0:
            index = 0
        else:
            index = int(math.floor((float(score) - float(table.minimum)) / table.bin_width))
        index = min(max(index, 0), table.log_tail.size - 1)
        return table.log_tail[index]
    raise ValueError(f"unknown table type: {type(table)!r}")


def transform_scores(table, scores):
    out = np.empty(scores.data.size, dtype=np.float32)
    parallel = use_parallel(scores.data.size)
    if isinstance(table, LogTailTable):
        kernel = transform_empirical_scores_parallel if parallel else transform_empirical_scores
        kernel(scores.data, table.scores, table.log_tail, out)
    elif isinstance(table, HybridLogTailTable):
        kernel = transform_hybrid_scores_parallel if parallel else transform_hybrid_scores
        kernel(
            scores.data,
            table.minimum,
            table.bin_width,
            table.log_tail,
            table.exact_tail.scores,
            table.exact_tail.log_tail,
            out,
        )
    else:
        raise ValueError(f"unknown table type: {type(table)!r}")
    return RaggedArray(out, scores.offsets.copy())


def flatten_bundle(bundle):
    fwd = bundle.forward.data
    rev = bundle.reverse.data
    if bundle.forward is bundle.reverse:
        return fwd.copy()
    return np.concatenate([fwd, rev])


def normalize_bundle(table, bundle):
    fwd = transform_scores(table, bundle.forward)
    if bundle.forward is bundle.reverse:
        return StrandPair(fwd, fwd)
    return StrandPair(fwd, transform_scores(table, bundle.reverse))


def _fit_normalize(strategy, raw, *, calibration=None, tail_logerr=0.0):
    if calibration is None:
        calibration = raw
    table = fit(strategy, flatten_bundle(calibration), tail_logerr=tail_logerr)
    return table, normalize_bundle(table, raw)
