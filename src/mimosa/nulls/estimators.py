"""Survival estimators for null distributions."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
from scipy.stats import gaussian_kde

from mimosa.nulls.types import NullDistributionData

logger = logging.getLogger(__name__)
MIN_KDE_SAMPLES = 3


class EmpiricalSurvivalEstimator:
    """Finite-sample-corrected empirical upper-tail survival estimator."""

    estimator_type = "ecdf"

    def __init__(self, scores: Iterable[float]):
        values = np.sort(np.asarray(list(scores), dtype=np.float64))
        if values.size == 0:
            raise ValueError("Null estimator requires at least one score.")
        self.scores = values

    @property
    def n(self) -> int:
        return int(self.scores.size)

    def pdf(self, score: float) -> float:
        return float(np.mean(np.isclose(self.scores, score)))

    def sf(self, score: float) -> float:
        count = self.n - int(np.searchsorted(self.scores, score, side="left"))
        return _clamp_pvalue((count + 1.0) / (self.n + 1.0), self.n)

    def to_entry(self) -> dict[str, Any]:
        return {
            "estimator_type": self.estimator_type,
            "sorted_scores": self.scores.astype(np.float64),
            "parameters": {},
        }


class KdeSurvivalEstimator(EmpiricalSurvivalEstimator):
    """Gaussian KDE upper-tail survival estimator with empirical p-value clamping."""

    estimator_type = "kde"

    def __init__(self, scores: Iterable[float]):
        super().__init__(scores)
        if self.n < MIN_KDE_SAMPLES or float(np.var(self.scores)) <= 0.0:
            raise ValueError("KDE requires at least three variable null scores.")
        self._kde = gaussian_kde(self.scores)

    def pdf(self, score: float) -> float:
        return float(self._kde.evaluate([score])[0])

    def sf(self, score: float) -> float:
        return _clamp_pvalue(float(self._kde.integrate_box_1d(score, np.inf)), self.n)

    def to_entry(self) -> dict[str, Any]:
        entry = super().to_entry()
        entry["estimator_type"] = self.estimator_type
        entry["parameters"] = {"bw_method": self._kde.factor}
        return entry


def fit_survival_estimator(scores: Iterable[float]) -> EmpiricalSurvivalEstimator:
    """Fit KDE when stable and otherwise fall back to an empirical estimator."""
    values = np.asarray(list(scores), dtype=np.float64)
    try:
        return KdeSurvivalEstimator(values)
    except Exception:
        return EmpiricalSurvivalEstimator(values)


def estimator_from_distribution(distribution: NullDistributionData) -> EmpiricalSurvivalEstimator:
    """Rehydrate an estimator from one null distribution file."""
    scores = np.asarray(distribution["sorted_scores"], dtype=np.float64)
    if distribution.get("estimator_type") == "kde":
        try:
            return KdeSurvivalEstimator(scores)
        except Exception:
            logger.warning("KDE null distribution could not be rehydrated; using ECDF.")
    return EmpiricalSurvivalEstimator(scores)


def _clamp_pvalue(value: float, n_null: int) -> float:
    lower = 1.0 / (int(n_null) + 1.0)
    return float(min(1.0, max(lower, value)))
