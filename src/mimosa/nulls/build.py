"""Null-distribution build workflow."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, cast

import numpy as np

from mimosa.cache import fingerprint_model
from mimosa.models import GenericModel
from mimosa.nulls.estimators import fit_survival_estimator
from mimosa.nulls.metadata import environment_metadata, stable_hash
from mimosa.nulls.storage import install_null_distribution_file, save_null_distribution_file
from mimosa.nulls.types import (
    NullBuildRequest,
    NullBuildResult,
    NullBuildSummary,
    NullDistributionData,
    NullDistributionFile,
    NullDistributionFileMetadata,
)
from mimosa.progress import iter_progress
from mimosa.types import ComparatorConfig

logger = logging.getLogger(__name__)


def build_null_distributions(  # noqa: PLR0913
    models: list[GenericModel],
    relations: dict[str, set[str]],
    *,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
    min_null_targets: int = 1,
    strict: bool = False,
    relation_fingerprint: str | None = None,
    progress: bool | None = False,
) -> NullBuildResult:
    """Build one pooled null distribution from all eligible query-target comparisons."""
    from mimosa.comparison import compare_one_to_many

    by_name = {model.name: model for model in models}
    collection_fp = stable_hash([fingerprint_model(model) for model in models])
    metadata_block = environment_metadata(
        strategy=strategy,
        config=config,
        sequences=sequences,
        background=background,
        model_collection_fingerprint=collection_fp,
        relation_fingerprint=relation_fingerprint,
    )
    skipped: list[dict[str, Any]] = []
    raw_scores: list[np.ndarray] = []
    included_query_names: list[str] = []
    included_target_names: set[str] = set()
    included_pairs: list[dict[str, str]] = []
    total_comparisons = 0
    score_only_config = replace(config, pvalue=False)

    for query in iter_progress(models, enabled=progress, desc="queries", total=len(models)):
        target_names = sorted(
            name for name in relations.get(query.name, set()) if name in by_name and name != query.name
        )
        if len(target_names) < min_null_targets:
            reason = f"only {len(target_names)} null target(s); required {min_null_targets}"
            skipped.append({"query": query.name, "reason": reason})
            message = f"Skipping null contribution for {query.name}: {reason}."
            if strict:
                raise ValueError(message)
            logger.warning(message)
            continue

        targets = [by_name[name] for name in target_names]
        results = compare_one_to_many(
            query,
            targets,
            strategy,
            score_only_config,
            sequences=sequences,
            background=background,
            progress=progress,
            progress_desc=f"{query.name} targets",
            progress_leave=False,
        )
        scores = np.asarray([float(result["score"]) for result in results], dtype=np.float64)
        raw_scores.append(scores)
        included_query_names.append(query.name)
        included_target_names.update(target_names)
        query_fingerprint = fingerprint_model(query)
        for target_name in target_names:
            included_pairs.append(
                {
                    "query_name": query.name,
                    "query_fingerprint": query_fingerprint,
                    "target_name": target_name,
                    "target_fingerprint": fingerprint_model(by_name[target_name]),
                }
            )
        total_comparisons += len(results)

    if not raw_scores:
        raise ValueError("Cannot build a null distribution: no eligible query-target comparisons were found.")

    all_scores = np.concatenate(raw_scores).astype(np.float64)
    estimator = fit_survival_estimator(all_scores)
    distribution = cast(NullDistributionData, estimator.to_entry())
    distribution.update(
        {
            "raw_null_scores": all_scores,
            "n_null": int(all_scores.size),
            "number_of_queries": len(included_query_names),
            "included_query_names": included_query_names,
            "included_target_names": sorted(included_target_names),
            "included_pairs": included_pairs,
        }
    )
    null_distribution_file: NullDistributionFile = {
        "metadata": cast(NullDistributionFileMetadata, metadata_block),
        "distribution": distribution,
    }
    return NullBuildResult(
        null_distribution_file=null_distribution_file,
        skipped=skipped,
        number_of_queries_used=len(included_query_names),
        total_comparisons=total_comparisons,
    )


def run_build_null_request(request: NullBuildRequest) -> NullBuildSummary:
    """Build, save, and optionally install one null distribution file."""
    built = build_null_distributions(
        request.models,
        request.relations,
        strategy=request.strategy,
        config=request.config,
        sequences=request.sequences,
        background=request.background,
        min_null_targets=request.min_null_targets,
        strict=request.strict,
        relation_fingerprint=request.relation_fingerprint,
        progress=request.progress,
    )
    null_distribution_file_path = save_null_distribution_file(built.null_distribution_file, request.output)
    cache_path = install_null_distribution_file(null_distribution_file_path) if request.install_cache else None
    return NullBuildSummary(
        null_distribution_file=null_distribution_file_path,
        cache_path=cache_path,
        number_of_motifs=len(request.models),
        number_of_queries_used=built.number_of_queries_used,
        skipped_queries=built.skipped,
        total_comparisons_run=built.total_comparisons,
        config_signature=built.null_distribution_file["metadata"]["config_signature"],
    )
