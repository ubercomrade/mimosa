"""Survival estimators for null distributions."""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import numpy as np
from scipy import stats

from mimosa.nulls.types import NullDistributionData

NUM_GENEXTREME_PARAMS = 3


class GenextremeSurvivalEstimator:
    """Generalized extreme value upper-tail survival estimator."""

    estimator_type = "genextreme"

    def __init__(self, scores: Iterable[float], params: Iterable[float] | None = None):
        values = np.sort(np.asarray(list(scores), dtype=np.float64))
        if values.size == 0:
            raise ValueError("Null estimator requires at least one score.")
        self.scores = values

        if params is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw_params = stats.genextreme.fit(self.scores)
        else:
            raw_params = params

        genextreme_params = tuple(float(value) for value in raw_params)
        if len(genextreme_params) != NUM_GENEXTREME_PARAMS:
            raise ValueError("GEV requires shape, location, and scale parameters.")
        if not np.all(np.isfinite(genextreme_params)) or genextreme_params[2] <= 0.0:
            raise ValueError("GEV parameters must be finite with positive scale.")
        self.genextreme_params = genextreme_params

    @property
    def n(self) -> int:
        return int(self.scores.size)

    def pdf(self, score: float) -> float:
        return float(stats.genextreme.pdf(score, *self.genextreme_params))

    def sf(self, score: float) -> float:
        return float(np.clip(stats.genextreme.sf(score, *self.genextreme_params), 0.0, 1.0))

    def to_entry(self) -> dict[str, Any]:
        return {
            "estimator_type": self.estimator_type,
            "genextreme_params": self.genextreme_params,
        }


def fit_survival_estimator(scores: Iterable[float]) -> GenextremeSurvivalEstimator:
    """Fit a GEV survival estimator."""
    return GenextremeSurvivalEstimator(scores)


def estimator_from_distribution(distribution: NullDistributionData) -> GenextremeSurvivalEstimator:
    """Rehydrate a GEV estimator from one null distribution file."""
    scores = _scores_from_distribution(distribution)
    parameters = distribution.get("parameters", {})
    raw_params: Iterable[float] | None
    if "genextreme_params" in distribution:
        raw_params = distribution["genextreme_params"]
    else:
        raw_params = parameters.get("genextreme_params")
    if raw_params is None:
        raise ValueError("Null distribution is missing genextreme_params.")
    return GenextremeSurvivalEstimator(scores, raw_params)


def _scores_from_distribution(distribution: NullDistributionData) -> np.ndarray:
    if "raw_null_scores" in distribution:
        return np.asarray(distribution["raw_null_scores"], dtype=np.float64)
    if "sorted_scores" in distribution:
        return np.asarray(distribution["sorted_scores"], dtype=np.float64)
    raise ValueError("Null distribution is missing raw_null_scores.")
