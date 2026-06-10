"""Comparison configuration helpers."""

from __future__ import annotations

from typing import Literal

from mimosa.types import ComparatorConfig
from mimosa.validation import (
    validate_cache_mode,
    validate_non_negative,
    validate_non_negative_int,
    validate_optional_thread_count,
    validate_pfm_top_fraction,
    validate_profile_normalization,
)

SUPPORTED_PROFILE_METRICS = ("co", "co_rowwise", "dice", "dice_rowwise", "cosine")
SUPPORTED_MOTIF_METRICS = ("pcc", "ed", "cosine")
MetricName = Literal["co", "co_rowwise", "dice", "dice_rowwise", "pcc", "ed", "cosine"]
_ALL_METRICS = frozenset((*SUPPORTED_PROFILE_METRICS, *SUPPORTED_MOTIF_METRICS))
SUPPORTED_PROFILE_NORMALIZATIONS = {"empirical_log_tail"}


def validate_metric(metric: str) -> MetricName:
    """Normalize and validate one public metric name."""
    normalized = str(metric).lower()
    if normalized not in _ALL_METRICS:
        options = ", ".join(sorted(_ALL_METRICS))
        raise ValueError(f"metric must be one of: {options}")
    return normalized  # type: ignore[return-value]


_validate_metric = validate_metric


def create_comparator_config(**kwargs) -> ComparatorConfig:
    """Build one validated immutable comparison config."""
    defaults = ComparatorConfig()
    allowed_keys = {field_name for field_name in defaults}
    unknown_keys = set(kwargs).difference(allowed_keys).difference({"promoters"})
    if unknown_keys:
        options = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown comparator option(s): {options}")

    config = defaults.to_dict()
    config.update(kwargs)
    legacy_background = config.pop("promoters", None)
    if "background" not in kwargs and legacy_background is not None:
        config["background"] = legacy_background
    config["metric"] = _validate_metric(config["metric"])
    config["min_logfpr"] = validate_non_negative("min_logfpr", config.get("min_logfpr"))
    config["window_radius"] = validate_non_negative_int("window_radius", config.get("window_radius", 10))
    config["realign_window"] = validate_non_negative_int("realign_window", config.get("realign_window", 3))
    config["profile_normalization"] = validate_profile_normalization(
        config.get("profile_normalization", "empirical_log_tail"),
        SUPPORTED_PROFILE_NORMALIZATIONS,
    )
    config["n_jobs"] = validate_optional_thread_count("n_jobs", config.get("n_jobs"))
    config["pfm_top_fraction"] = validate_pfm_top_fraction(config.get("pfm_top_fraction")) or defaults.pfm_top_fraction
    config["cache_mode"] = validate_cache_mode(config.get("cache_mode", "off"))
    null_search_dirs = config.get("null_search_dirs")
    if null_search_dirs is not None:
        config["null_search_dirs"] = tuple(null_search_dirs)
    return ComparatorConfig(**config)
