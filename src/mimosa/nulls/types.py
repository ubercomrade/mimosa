"""Typed null-distribution payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from mimosa.batches import SequenceBatch
from mimosa.models import GenericModel
from mimosa.types import ComparatorConfig


class NullDistributionFileMetadata(TypedDict):
    format_version: int
    created_at: str
    strategy: str
    metric: str
    sequence_fingerprint: str
    background_fingerprint: str
    model_collection_fingerprint: str | None
    relation_fingerprint: str | None
    package_version: str


class NullDistributionData(TypedDict, total=False):
    estimator_type: str
    sorted_scores: np.ndarray
    parameters: dict[str, Any]
    genextreme_params: tuple[float, float, float]
    raw_null_scores: np.ndarray
    n_null: int
    number_of_queries: int
    included_query_names: list[str]
    included_target_names: list[str]
    included_pairs: list[dict[str, str]]


class NullDistributionFile(TypedDict):
    metadata: NullDistributionFileMetadata
    distribution: NullDistributionData


@dataclass
class NullBuildResult:
    """Summary returned after building one in-memory null distribution file payload."""

    null_distribution_file: NullDistributionFile
    skipped: list[dict[str, Any]]
    number_of_queries_used: int
    total_comparisons: int


@dataclass(frozen=True, slots=True)
class NullBuildRequest:
    """Resolved inputs required to build and persist one null distribution file."""

    models: list[GenericModel]
    relations: dict[str, set[str]]
    strategy: str
    config: ComparatorConfig
    output: str | Path
    sequences: SequenceBatch | None = None
    background: SequenceBatch | None = None
    min_null_targets: int = 1
    strict: bool = False
    relation_fingerprint: str | None = None
    install_cache: bool = False
    progress: bool | None = False


@dataclass(frozen=True, slots=True)
class NullBuildSummary:
    """Serializable summary for one null-distribution build."""

    null_distribution_file: Path
    cache_path: Path | None
    number_of_motifs: int
    number_of_queries_used: int
    skipped_queries: list[dict[str, Any]]
    total_comparisons_run: int

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON-compatible payload used by the CLI."""
        return {
            "null_distribution_file": str(self.null_distribution_file),
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "number_of_motifs": self.number_of_motifs,
            "number_of_queries_used": self.number_of_queries_used,
            "skipped_queries": self.skipped_queries,
            "total_comparisons_run": self.total_comparisons_run,
        }
