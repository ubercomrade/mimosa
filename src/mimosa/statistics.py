"""Statistics for pair-specific Monte Carlo results."""

from dataclasses import dataclass

import numpy as np

from .compare import ComparisonResult


@dataclass(frozen=True)
class AnnotatedResult:
    result: ComparisonResult
    p_value: float | None = None
    adj_p_value: float | None = None
    e_value: float | None = None
    mc_replicates: int | None = None
    mc_exceedances: int | None = None
    mc_seed: int | None = None
    mc_group: str | None = None

    def __getattr__(self, name):
        if name == "result" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.result, name)

    def to_dict(self):
        d = self.result.to_dict()
        d["annotation_schema_version"] = 2
        if self.p_value is not None:
            d["p-value"] = self.p_value
        if self.adj_p_value is not None:
            d["adj.p-value"] = self.adj_p_value
        if self.e_value is not None:
            d["E-value"] = self.e_value
        if self.mc_replicates is not None:
            d["monte_carlo"] = {
                "replicates": self.mc_replicates, "exceedances": self.mc_exceedances,
                "seed": self.mc_seed, "group": self.mc_group,
                "version": "mimosa-direct-mc-v1",
            }
        return d


def adjusted_pvalues(pvalues):
    """Benjamini–Hochberg adjustment across the supplied hypothesis family."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.array([], dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and lie in [0, 1].")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    adj = np.minimum.accumulate((sorted_p * n / ranks)[::-1])[::-1]
    result = np.empty(n, dtype=np.float64)
    result[order] = np.minimum(adj, 1.0)
    return result


def evalue(pvalue, effective_n):
    p = float(pvalue)
    if not (np.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError("p-value must be finite and lie in [0, 1].")
    if isinstance(effective_n, bool) or not isinstance(effective_n, (int, float, np.integer, np.floating)):
        raise TypeError("effective_n must be a real number.")
    if not (np.isfinite(effective_n) and effective_n >= 0):
        raise ValueError("effective_n must be finite and non-negative.")
    return p * effective_n


def empirical_upper_tail_pvalue(scores, score):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("null scores must be nonempty and finite.")
    if not np.isfinite(score):
        raise ValueError("score must be finite.")
    return (np.count_nonzero(scores >= score) + 1) / (scores.size + 1)
