"""Null-distribution compatibility lookup and validation."""

from __future__ import annotations

from pathlib import Path

from mimosa.models import GenericModel
from mimosa.nulls.metadata import environment_metadata
from mimosa.nulls.storage import NULL_CACHE_DIR, load_null_distribution_file
from mimosa.nulls.types import NullDistributionFile
from mimosa.types import ComparatorConfig


def load_compatible_null_distribution_file(
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> NullDistributionFile | None:
    """Load the explicit or first searched compatible null distribution file for one query/config."""
    explicit = config.get("null_distribution")
    if explicit is not None:
        null_distribution_file = load_null_distribution_file(explicit)
        validate_null_distribution_file_compatible(
            null_distribution_file,
            strategy=strategy,
            config=config,
            query_model=query_model,
            sequences=sequences,
            background=background,
        )
        return null_distribution_file

    search_dirs = [Path(p) for p in (config.get("null_search_dirs") or [])]
    search_dirs.append(NULL_CACHE_DIR)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.joblib")):
            null_distribution_file = load_null_distribution_file(path)
            if is_null_distribution_file_compatible(
                null_distribution_file,
                strategy=strategy,
                config=config,
                query_model=query_model,
                sequences=sequences,
                background=background,
            ):
                return null_distribution_file
    return None


def validate_null_distribution_file_compatible(  # noqa: PLR0913
    null_distribution_file: NullDistributionFile,
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> None:
    """Raise ValueError if an explicit null distribution file does not match the comparison context."""
    problems = _compatibility_problems(
        null_distribution_file,
        strategy=strategy,
        config=config,
        query_model=query_model,
        sequences=sequences,
        background=background,
    )
    if problems:
        raise ValueError("Null distribution is incompatible: " + "; ".join(problems))


def is_null_distribution_file_compatible(  # noqa: PLR0913
    null_distribution_file: NullDistributionFile,
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> bool:
    """Return True when one null distribution file matches the comparison context."""
    return not _compatibility_problems(
        null_distribution_file,
        strategy=strategy,
        config=config,
        query_model=query_model,
        sequences=sequences,
        background=background,
    )


def _compatibility_problems(  # noqa: PLR0913
    null_distribution_file: NullDistributionFile,
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> list[str]:
    del query_model
    metadata_block = null_distribution_file.get("metadata", {})
    expected = environment_metadata(strategy=strategy, config=config, sequences=sequences, background=background)
    problems: list[str] = []
    compatibility_keys = (
        "format_version",
        "strategy",
        "metric",
        "sequence_fingerprint",
        "background_fingerprint",
    )
    for key in compatibility_keys:
        if metadata_block.get(key) != expected.get(key):
            problems.append(f"{key} differs")

    if "distribution" not in null_distribution_file:
        problems.append("pooled null distribution is missing")
    else:
        distribution = null_distribution_file["distribution"]
        parameters = distribution.get("parameters", {})
        if "genextreme_params" not in distribution and "genextreme_params" not in parameters:
            problems.append("genextreme_params are missing")
    return problems
