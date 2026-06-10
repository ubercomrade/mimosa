"""Result annotation with null-distribution significance values."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

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
    """Return comparison results enriched with p-value, E-value, and BH-FDR q-value."""
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

    for idx, qvalue in zip(valid_indices, bh_qvalues(pvalues), strict=False):
        annotated_results[idx] = replace(annotated_results[idx], q_value=qvalue)
    return annotated_results


def bh_qvalues(pvalues: Iterable[float]) -> list[float]:
    """Compute monotone Benjamini-Hochberg q-values preserving input order."""
    values = np.asarray(list(pvalues), dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = ranked.size
    for reverse_idx in range(m - 1, -1, -1):
        rank = reverse_idx + 1
        running = min(running, float(ranked[reverse_idx] * m / rank))
        adjusted[reverse_idx] = running
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(value) for value in result]
