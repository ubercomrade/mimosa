"""Result annotation with null-distribution significance values."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import stats

from mimosa.models import GenericModel
from mimosa.nulls.estimators import estimator_from_distribution
from mimosa.nulls.metadata import stable_hash
from mimosa.nulls.types import NullDistributionFile
from mimosa.types import ComparisonResult


def annotate_results_with_nulls(
    results: list[ComparisonResult],
    *,
    null_distribution_file: NullDistributionFile,
    query_model: GenericModel,
    effective_number_of_targets: int | None = None,
) -> list[ComparisonResult]:
    """Return comparison results enriched with p-value, adjusted p-value, and E-value."""
    del query_model
    distribution = null_distribution_file["distribution"]
    estimator = estimator_from_distribution(distribution)
    n_null = int(distribution.get("n_null", estimator.n))
    metadata_block = null_distribution_file.get("metadata", {})
    null_id = stable_hash(
        {
            "format_version": metadata_block.get("format_version"),
            "config_signature_hash": metadata_block.get("config_signature_hash"),
            "model_collection_fingerprint": metadata_block.get("model_collection_fingerprint"),
            "relation_fingerprint": metadata_block.get("relation_fingerprint"),
            "n_null": n_null,
        }
    )
    effective = effective_number_of_targets or len(results)

    pvalues: list[float] = []
    annotated_results = list(results)
    valid_indices: list[int] = []
    for idx, result in enumerate(results):
        if "score" not in result:
            continue
        pvalue = estimator.sf(float(result["score"]))
        annotated_results[idx] = replace(
            result,
            p_value=pvalue,
            e_value=float(pvalue * effective),
            null_id=null_id,
            null_n=n_null,
            null_estimator=str(distribution.get("estimator_type", estimator.estimator_type)),
        )
        pvalues.append(pvalue)
        valid_indices.append(idx)

    for idx, adjusted_pvalue in zip(valid_indices, adjusted_pvalues(pvalues), strict=False):
        annotated_results[idx] = replace(annotated_results[idx], adj_p_value=adjusted_pvalue)
    return annotated_results


def adjusted_pvalues(pvalues) -> list[float]:
    """Compute FDR-adjusted p-values preserving input order."""
    values = np.asarray(list(pvalues), dtype=np.float64)
    if values.size == 0:
        return []
    if values.size == 1:
        return [float(values[0])]
    result = stats.false_discovery_control(values, method="bh")
    return [float(value) for value in result]
