"""Functional motif comparison workflows."""

from __future__ import annotations

from mimosa.comparison.config import (
    SUPPORTED_MOTIF_METRICS,
    SUPPORTED_PROFILE_METRICS,
    create_comparator_config,
    validate_metric,
)
from mimosa.comparison.motif import strategy_motif
from mimosa.comparison.profile import strategy_profile
from mimosa.comparison.runner import compare, compare_one_to_many, registry

__all__ = [
    "SUPPORTED_MOTIF_METRICS",
    "SUPPORTED_PROFILE_METRICS",
    "compare",
    "compare_one_to_many",
    "create_comparator_config",
    "registry",
    "strategy_motif",
    "strategy_profile",
    "validate_metric",
]
