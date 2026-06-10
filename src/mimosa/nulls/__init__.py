"""Distribution-backed significance helpers."""

from __future__ import annotations

from mimosa.nulls.annotation import annotate_results_with_nulls, bh_qvalues
from mimosa.nulls.build import build_null_distributions, run_build_null_request
from mimosa.nulls.compatibility import (
    is_null_distribution_file_compatible,
    load_compatible_null_distribution_file,
    validate_null_distribution_file_compatible,
)
from mimosa.nulls.estimators import EmpiricalSurvivalEstimator, KdeSurvivalEstimator, fit_survival_estimator
from mimosa.nulls.metadata import (
    NULL_FORMAT_VERSION,
    comparator_signature,
    environment_metadata,
    file_fingerprint,
    package_metadata,
    stable_hash,
    stable_json_dumps,
)
from mimosa.nulls.relations import parse_group_relations, parse_pair_matrix_relations, parse_pair_relations
from mimosa.nulls.storage import (
    NULL_CACHE_DIR,
    install_null_distribution_file,
    load_null_distribution_file,
    save_null_distribution_file,
)
from mimosa.nulls.types import (
    NullBuildRequest,
    NullBuildResult,
    NullBuildSummary,
    NullDistributionData,
    NullDistributionFile,
    NullDistributionFileMetadata,
)

__all__ = [
    "NULL_CACHE_DIR",
    "NULL_FORMAT_VERSION",
    "EmpiricalSurvivalEstimator",
    "KdeSurvivalEstimator",
    "NullBuildRequest",
    "NullBuildResult",
    "NullBuildSummary",
    "NullDistributionData",
    "NullDistributionFile",
    "NullDistributionFileMetadata",
    "annotate_results_with_nulls",
    "bh_qvalues",
    "build_null_distributions",
    "comparator_signature",
    "environment_metadata",
    "file_fingerprint",
    "fit_survival_estimator",
    "install_null_distribution_file",
    "is_null_distribution_file_compatible",
    "load_compatible_null_distribution_file",
    "load_null_distribution_file",
    "parse_group_relations",
    "parse_pair_matrix_relations",
    "parse_pair_relations",
    "package_metadata",
    "run_build_null_request",
    "save_null_distribution_file",
    "stable_hash",
    "stable_json_dumps",
    "validate_null_distribution_file_compatible",
]
