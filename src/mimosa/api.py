"""High-level public API for motif comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from mimosa.batches import SequenceBatch, make_random_sequence_batch
from mimosa.comparison import (
    SUPPORTED_MOTIF_METRICS,
    SUPPORTED_PROFILE_METRICS,
    compare,
    create_comparator_config,
)
from mimosa.comparison import (
    compare_one_to_many as compare_one_to_many_models,
)
from mimosa.io import read_fasta
from mimosa.models import GenericModel, read_model, read_models
from mimosa.nulls import (
    NullBuildRequest,
    NullBuildSummary,
    annotate_results_with_nulls,
    file_fingerprint,
    load_compatible_null_distribution_file,
    parse_group_relations,
    run_build_null_request,
)
from mimosa.types import ComparatorConfig, ComparisonResult, OneToManyConfig, OneToOneConfig
from mimosa.validation import validate_file_exists, validate_positive_int

logger = logging.getLogger(__name__)

ModelRef = GenericModel | str | Path
ModelCollectionRef = Iterable[ModelRef] | str | Path
SequenceRef = SequenceBatch | str | Path

_STRATEGY_ALIASES = {
    "profile": "profile",
    "motif": "motif",
}

_DEFAULT_METRICS = {
    "profile": "co",
    "motif": "pcc",
}

_ALLOWED_METRICS = {
    "profile": frozenset(SUPPORTED_PROFILE_METRICS),
    "motif": frozenset(SUPPORTED_MOTIF_METRICS),
}


def create_one_to_one_config(
    query: ModelRef,
    target: ModelRef,
    query_type: str | None = None,
    target_type: str | None = None,
    strategy: str = "profile",
    sequences: SequenceRef | None = None,
    background: SequenceRef | None = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: ComparatorConfig | None = None,
    query_kwargs: Mapping[str, Any] | None = None,
    target_kwargs: Mapping[str, Any] | None = None,
    **comparator_kwargs: Any,
) -> OneToOneConfig:
    """Build a unified immutable one-vs-one comparison configuration."""
    normalized_strategy = _normalize_strategy(strategy)
    if comparator is not None and comparator_kwargs:
        raise ValueError("Use either 'comparator' or comparator kwargs, not both.")

    effective_kwargs = dict(comparator_kwargs)
    default_metric = _DEFAULT_METRICS.get(normalized_strategy)
    if comparator is None and default_metric is not None and "metric" not in effective_kwargs:
        effective_kwargs["metric"] = default_metric

    resolved_comparator = comparator or create_comparator_config(**effective_kwargs)
    resolved_query_kwargs = dict(query_kwargs or {})
    resolved_target_kwargs = dict(target_kwargs or {})
    return OneToOneConfig(
        query=query,
        target=target,
        query_type=query_type,
        target_type=target_type,
        strategy=normalized_strategy,
        sequences=sequences,
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=resolved_comparator,
        query_kwargs=resolved_query_kwargs,
        target_kwargs=resolved_target_kwargs,
    )


def compare_one_to_one(
    query: ModelRef,
    target: ModelRef,
    query_type: str | None = None,
    target_type: str | None = None,
    strategy: str = "profile",
    sequences: SequenceRef | None = None,
    background: SequenceRef | None = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: ComparatorConfig | None = None,
    query_kwargs: Mapping[str, Any] | None = None,
    target_kwargs: Mapping[str, Any] | None = None,
    **comparator_kwargs: Any,
) -> ComparisonResult:
    """Single-call entry point for one-vs-one motif comparison."""
    config = create_one_to_one_config(
        query=query,
        target=target,
        query_type=query_type,
        target_type=target_type,
        strategy=strategy,
        sequences=sequences,
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=comparator,
        query_kwargs=query_kwargs,
        target_kwargs=target_kwargs,
        **comparator_kwargs,
    )
    return run_one_to_one(config)


def create_one_to_many_config(
    query: ModelRef,
    targets: list[ModelRef],
    query_type: str | None = None,
    target_type: str | None = None,
    strategy: str = "profile",
    sequences: SequenceRef | None = None,
    background: SequenceRef | None = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: ComparatorConfig | None = None,
    query_kwargs: Mapping[str, Any] | None = None,
    target_kwargs: Mapping[str, Any] | None = None,
    progress: bool | None = False,
    **comparator_kwargs: Any,
) -> OneToManyConfig:
    """Build a unified immutable one-vs-many comparison configuration."""
    normalized_strategy = _normalize_strategy(strategy)
    if comparator is not None and comparator_kwargs:
        raise ValueError("Use either 'comparator' or comparator kwargs, not both.")

    effective_kwargs = dict(comparator_kwargs)
    default_metric = _DEFAULT_METRICS.get(normalized_strategy)
    if comparator is None and default_metric is not None and "metric" not in effective_kwargs:
        effective_kwargs["metric"] = default_metric

    resolved_comparator = comparator or create_comparator_config(**effective_kwargs)
    resolved_query_kwargs = dict(query_kwargs or {})
    resolved_target_kwargs = dict(target_kwargs or {})
    return OneToManyConfig(
        query=query,
        targets=tuple(_normalize_targets(targets)),
        query_type=query_type,
        target_type=target_type,
        strategy=normalized_strategy,
        sequences=sequences,
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=resolved_comparator,
        query_kwargs=resolved_query_kwargs,
        target_kwargs=resolved_target_kwargs,
        progress=progress,
    )


def compare_one_to_many(
    query: ModelRef,
    targets: list[ModelRef],
    query_type: str | None = None,
    target_type: str | None = None,
    strategy: str = "profile",
    sequences: SequenceRef | None = None,
    background: SequenceRef | None = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: ComparatorConfig | None = None,
    query_kwargs: Mapping[str, Any] | None = None,
    target_kwargs: Mapping[str, Any] | None = None,
    progress: bool | None = False,
    **comparator_kwargs: Any,
) -> list[ComparisonResult]:
    """Single-call entry point for one-vs-many motif comparison."""
    config = create_one_to_many_config(
        query=query,
        targets=targets,
        query_type=query_type,
        target_type=target_type,
        strategy=strategy,
        sequences=sequences,
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=comparator,
        query_kwargs=query_kwargs,
        target_kwargs=target_kwargs,
        progress=progress,
        **comparator_kwargs,
    )
    return run_one_to_many(config)


def create_null_distribution_config(  # noqa: PLR0913
    models: ModelCollectionRef,
    *,
    output: str | Path,
    groups: str | Path,
    model_type: str | None = None,
    pattern: str | None = None,
    name_column: str = "motif",
    group_column: str = "group",
    ignore_missing_relations: bool = False,
    strategy: str = "motif",
    sequences: SequenceRef | None = None,
    background: SequenceRef | None = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: ComparatorConfig | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    min_null_targets: int = 1,
    strict: bool = False,
    install_cache: bool = False,
    progress: bool | None = False,
    **comparator_kwargs: Any,
) -> NullBuildRequest:
    """Build a resolved null-distribution request for interactive API use."""
    normalized_strategy = _normalize_strategy(strategy)
    if comparator is not None and comparator_kwargs:
        raise ValueError("Use either 'comparator' or comparator kwargs, not both.")

    effective_kwargs = dict(comparator_kwargs)
    default_metric = _DEFAULT_METRICS.get(normalized_strategy)
    if comparator is None and default_metric is not None and "metric" not in effective_kwargs:
        effective_kwargs["metric"] = default_metric
    resolved_comparator = comparator or create_comparator_config(**effective_kwargs)
    _validate_comparator_for_strategy(normalized_strategy, resolved_comparator)

    resolved_models = _resolve_model_collection(models, model_type, pattern, model_kwargs)
    known_names = {model.name for model in resolved_models}
    resolved_relations, relation_fingerprint = _resolve_null_relations(
        groups=groups,
        known_names=known_names,
        ignore_missing=ignore_missing_relations,
        name_column=name_column,
        group_column=group_column,
    )
    resolved_sequences = _resolve_null_build_sequences(
        sequences,
        strategy=normalized_strategy,
        comparator=resolved_comparator,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
    )
    resolved_background = _resolve_sequence_ref(background, num_sequences, seq_length, seed) if background else None

    return NullBuildRequest(
        models=resolved_models,
        relations=resolved_relations,
        strategy=normalized_strategy,
        config=resolved_comparator,
        output=output,
        sequences=resolved_sequences,
        background=resolved_background,
        min_null_targets=validate_positive_int("min_null_targets", min_null_targets),
        strict=strict,
        relation_fingerprint=relation_fingerprint,
        install_cache=install_cache,
        progress=progress,
    )


def create_null_distribution(  # noqa: PLR0913
    models: ModelCollectionRef,
    *,
    output: str | Path,
    groups: str | Path,
    model_type: str | None = None,
    pattern: str | None = None,
    name_column: str = "motif",
    group_column: str = "group",
    ignore_missing_relations: bool = False,
    strategy: str = "motif",
    sequences: SequenceRef | None = None,
    background: SequenceRef | None = None,
    num_sequences: int = 1000,
    seq_length: int = 200,
    seed: int = 127,
    comparator: ComparatorConfig | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    min_null_targets: int = 1,
    strict: bool = False,
    install_cache: bool = False,
    progress: bool | None = False,
    **comparator_kwargs: Any,
) -> NullBuildSummary:
    """Build and persist one null distribution from model and relation inputs."""
    config = create_null_distribution_config(
        models=models,
        output=output,
        model_type=model_type,
        pattern=pattern,
        groups=groups,
        name_column=name_column,
        group_column=group_column,
        ignore_missing_relations=ignore_missing_relations,
        strategy=strategy,
        sequences=sequences,
        background=background,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        comparator=comparator,
        model_kwargs=model_kwargs,
        min_null_targets=min_null_targets,
        strict=strict,
        install_cache=install_cache,
        progress=progress,
        **comparator_kwargs,
    )
    return run_null_distribution(config)


def run_null_distribution(config: NullBuildRequest) -> NullBuildSummary:
    """Execute a resolved null-distribution build request."""
    return run_build_null_request(config)


def run_one_to_one(config: OneToOneConfig) -> ComparisonResult:
    """Execute one comparison from a one-vs-one config."""
    strategy = _normalize_strategy(config.strategy)
    query_model = _resolve_model(config.query, config.query_type, config.query_kwargs)
    target_model = _resolve_model(config.target, config.target_type, config.target_kwargs)
    _validate_models_for_strategy(strategy, query_model, target_model)
    _validate_comparator_for_strategy(strategy, config.comparator)

    background = _resolve_optional_sequences(config.background, config)
    needs_sequences = _needs_sequences(strategy, config.comparator, query_model, target_model)
    sequences = _resolve_sequences(config.sequences, config) if needs_sequences else None
    result = compare(
        model1=query_model,
        model2=target_model,
        strategy=strategy,
        config=config.comparator,
        sequences=sequences,
        background=background,
    )
    return _annotate_results_if_requested(
        [result],
        query_model=query_model,
        strategy=strategy,
        config=config.comparator,
        sequences=sequences,
        background=background,
        default_effective_number_of_targets=1,
    )[0]


def run_one_to_many(config: OneToManyConfig) -> list[ComparisonResult]:
    """Execute one comparison of a single query against many targets."""
    strategy = _normalize_strategy(config.strategy)
    query_model = _resolve_model(config.query, config.query_type, config.query_kwargs)
    _validate_comparator_for_strategy(strategy, config.comparator)

    if not config.targets:
        return []

    background = _resolve_optional_sequences(config.background, config)
    target_models = _resolve_target_models(
        config.targets,
        config.target_type,
        config.target_kwargs,
        strategy,
        query_model,
    )
    needs_sequences = any(
        _needs_sequences(strategy, config.comparator, query_model, target_model) for target_model in target_models
    )
    sequences = _resolve_sequences(config.sequences, config) if needs_sequences else None
    results = compare_one_to_many_models(
        query_model=query_model,
        target_models=iter(target_models),
        strategy=strategy,
        config=config.comparator,
        sequences=sequences,
        background=background,
        progress=config.progress,
        progress_desc=f"{query_model.name} targets",
    )
    return _annotate_results_if_requested(
        results,
        query_model=query_model,
        strategy=strategy,
        config=config.comparator,
        sequences=sequences,
        background=background,
        default_effective_number_of_targets=len(results),
    )


def _normalize_strategy(strategy: str) -> str:
    """Normalize strategy aliases to internal names."""
    resolved = _STRATEGY_ALIASES.get(strategy.lower())
    if resolved is None:
        available = ", ".join(sorted(_STRATEGY_ALIASES))
        raise ValueError(f"Unknown strategy: {strategy!r}. Available: {available}")
    return resolved


def _resolve_model(model: ModelRef, model_type: str | None, kwargs: Mapping[str, Any]) -> GenericModel:
    """Convert one model reference to GenericModel."""
    if isinstance(model, GenericModel):
        return model
    if isinstance(model, (str, Path)):
        if model_type is None:
            raise ValueError("model_type is required when model is provided as a file path.")
        return read_model(str(model), model_type, **dict(kwargs))
    raise TypeError(f"Unsupported model reference type: {type(model)!r}")


def _resolve_model_collection(
    models: ModelCollectionRef,
    model_type: str | None,
    pattern: str | None,
    kwargs: Mapping[str, Any] | None,
) -> list[GenericModel]:
    """Resolve a motif collection from in-memory models, paths, or a collection source."""
    effective_kwargs = dict(kwargs or {})
    if isinstance(models, (str, Path)):
        if model_type is None:
            raise ValueError("model_type is required when models are provided as a collection path.")
        return read_models(models, model_type, pattern=pattern, **effective_kwargs)

    if isinstance(models, GenericModel):
        raise TypeError("models must be a collection of model references, not a single model.")

    resolved = [_resolve_model(model, model_type, effective_kwargs) for model in models]
    if not resolved:
        raise ValueError("models must contain at least one model.")
    return resolved


def _normalize_targets(targets: Iterable[ModelRef]) -> list[ModelRef]:
    """Normalize one targets collection and reject scalar inputs."""
    if isinstance(targets, (str, Path, GenericModel)):
        raise TypeError("targets must be a list of model references, not a single model.")
    return list(targets)


def _resolve_target_models(
    targets: Iterable[ModelRef],
    model_type: str | None,
    kwargs: Mapping[str, Any],
    strategy: str,
    query_model: GenericModel,
) -> tuple[GenericModel, ...]:
    """Resolve and validate target models once for one-vs-many comparison."""
    resolved_targets: list[GenericModel] = []
    for target in targets:
        target_model = _resolve_model(target, model_type, kwargs)
        _validate_models_for_strategy(strategy, query_model, target_model)
        resolved_targets.append(target_model)
    return tuple(resolved_targets)


def _needs_sequences(strategy: str, comparator: ComparatorConfig, model1: GenericModel, model2: GenericModel) -> bool:
    """Return True if the selected strategy requires sequence input."""
    if strategy == "profile":
        return model1.type_key != "scores" or model2.type_key != "scores"
    return strategy == "motif" and (comparator["pfm_mode"] or model1.type_key != model2.type_key)


def _validate_models_for_strategy(strategy: str, model1: GenericModel, model2: GenericModel) -> None:
    """Validate model combinations for the selected comparison strategy."""
    if strategy == "motif" and ("scores" in {model1.type_key, model2.type_key}):
        raise ValueError("Motif strategy does not support score-profile inputs.")


def _validate_comparator_for_strategy(strategy: str, comparator: ComparatorConfig) -> None:
    """Validate comparator options for the selected strategy."""
    allowed_metrics = _ALLOWED_METRICS.get(strategy)
    if allowed_metrics is not None and comparator["metric"] not in allowed_metrics:
        options = ", ".join(sorted(allowed_metrics))
        raise ValueError(f"Strategy '{strategy}' requires one of the following metrics: {options}")


def _resolve_optional_sequences(
    source: SequenceRef | None,
    config: OneToOneConfig | OneToManyConfig,
) -> SequenceBatch | None:
    """Resolve optional sequence input while preserving explicit None."""
    if source is None:
        return None
    return _resolve_sequences(source, config)


def _resolve_sequences(source: SequenceRef | None, config: OneToOneConfig | OneToManyConfig) -> SequenceBatch:
    """Resolve one sequence source to a padded sequence batch."""
    if source is None:
        num_sequences = validate_positive_int("num_sequences", config.num_sequences)
        seq_length = validate_positive_int("seq_length", config.seq_length)
        return make_random_sequence_batch(num_sequences, seq_length, config.seed)
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        path = validate_file_exists(source, "Sequence file")
        return read_fasta(path)
    raise TypeError(f"Unsupported sequence source type: {type(source)!r}")


def _resolve_sequence_ref(
    source: SequenceRef | None,
    num_sequences: int,
    seq_length: int,
    seed: int,
) -> SequenceBatch:
    """Resolve one sequence source for null-distribution building."""
    if source is None:
        resolved_num_sequences = validate_positive_int("num_sequences", num_sequences)
        resolved_seq_length = validate_positive_int("seq_length", seq_length)
        return make_random_sequence_batch(resolved_num_sequences, resolved_seq_length, seed)
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        path = validate_file_exists(source, "Sequence file")
        return read_fasta(path)
    raise TypeError(f"Unsupported sequence source type: {type(source)!r}")


def _resolve_null_build_sequences(
    source: SequenceRef | None,
    *,
    strategy: str,
    comparator: ComparatorConfig,
    num_sequences: int,
    seq_length: int,
    seed: int,
) -> SequenceBatch | None:
    """Resolve sequences only when the null-build scoring path needs them."""
    needs_sequences = strategy == "profile" or bool(comparator["pfm_mode"])
    if not needs_sequences:
        return None
    return _resolve_sequence_ref(source, num_sequences, seq_length, seed)


def _resolve_null_relations(
    *,
    groups: str | Path,
    known_names: set[str],
    ignore_missing: bool,
    name_column: str,
    group_column: str,
) -> tuple[dict[str, set[str]], str]:
    """Resolve null-pair relations from a motif-to-group table."""
    path = validate_file_exists(groups, "Group relation file")
    return (
        parse_group_relations(
            path,
            name_column=name_column,
            group_column=group_column,
            ignore_missing=ignore_missing,
            known_names=known_names,
        ),
        file_fingerprint(path),
    )


def _annotate_results_if_requested(
    results: list[ComparisonResult],
    *,
    query_model: GenericModel,
    strategy: str,
    config: ComparatorConfig,
    sequences: SequenceBatch | None = None,
    background: SequenceBatch | None = None,
    default_effective_number_of_targets: int,
) -> list[ComparisonResult]:
    """Resolve significance metadata on the API boundary when requested."""
    if not config.pvalue or not results:
        return results

    null_distribution_file = load_compatible_null_distribution_file(
        strategy=strategy,
        config=config,
        query_model=query_model,
        sequences=sequences,
        background=background,
    )
    if null_distribution_file is None:
        logger.warning("No compatible null distribution found; returning score-only result.")
        return results

    effective_number = config.effective_number_of_targets or default_effective_number_of_targets
    return annotate_results_with_nulls(
        results,
        null_distribution_file=null_distribution_file,
        query_model=query_model,
        effective_number_of_targets=effective_number,
    )
