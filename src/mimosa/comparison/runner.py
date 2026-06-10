"""Public comparison dispatch and one-to-many execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, TypeVar

from joblib import Parallel, delayed

from mimosa.comparison.motif import _compare_motif_one_to_many, strategy_motif
from mimosa.comparison.profile import _compare_profile_one_to_many, strategy_profile
from mimosa.models import GenericModel
from mimosa.types import ComparatorConfig, ComparisonResult

_ComparisonJobResult = TypeVar("_ComparisonJobResult")

registry: dict[str, Callable] = {
    "motif": strategy_motif,
    "profile": strategy_profile,
}


def _register_comparison_strategy(name: str):
    """Register one comparison strategy."""

    def decorator(fn):
        registry[name] = fn
        return fn

    return decorator


def _resolve_target_job_count(n_jobs: int | None) -> int:
    """Resolve one target-level worker count from the compatibility config key."""
    return -1 if n_jobs is None else int(n_jobs)


def _run_target_comparisons(
    target_models: list[GenericModel],
    n_jobs: int | None,
    worker: Callable[[GenericModel], _ComparisonJobResult],
) -> list[_ComparisonJobResult]:
    """Execute one worker across targets sequentially or with joblib threads."""
    if not target_models:
        return []

    n_jobs = _resolve_target_job_count(n_jobs)
    if n_jobs == 1 or len(target_models) == 1:
        return [worker(target_model) for target_model in target_models]

    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(worker)(target_model) for target_model in target_models)


def compare(
    model1: GenericModel,
    model2: GenericModel,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
) -> ComparisonResult:
    """Main entry point for motif comparison."""
    try:
        strategy_fn = registry[strategy]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}") from exc

    effective_config = replace(config, background=background) if background is not None else config
    return strategy_fn(model1, model2, sequences, effective_config)


def compare_one_to_many(
    query_model: GenericModel,
    target_models,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
) -> list[ComparisonResult]:
    """Main entry point for one-vs-many motif comparison."""
    effective_config = replace(config, background=background) if background is not None else config

    if strategy == "profile":
        return _compare_profile_one_to_many(query_model, target_models, sequences, effective_config)
    if strategy == "motif":
        return _compare_motif_one_to_many(query_model, target_models, sequences, effective_config)
    available = ", ".join(sorted(registry))
    raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")
