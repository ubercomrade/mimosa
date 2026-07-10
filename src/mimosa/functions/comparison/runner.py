"""Public comparison dispatch and one-to-many execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

from numba import config as numba_config
from numba import get_num_threads, set_num_threads

from mimosa.comparison.motif import _compare_motif_one_to_many, strategy_motif
from mimosa.comparison.profile import _compare_profile_one_to_many, strategy_profile
from mimosa.models import GenericModel
from mimosa.progress import iter_progress
from mimosa.types import ComparatorConfig, ComparisonResult

_ComparisonJobResult = TypeVar("_ComparisonJobResult")

registry: dict[str, Callable] = {
    "motif": strategy_motif,
    "profile": strategy_profile,
}


def _resolve_numba_thread_count(n_jobs: int | None) -> int | None:
    """Resolve the compatibility option against Numba's runtime maximum."""
    if n_jobs is None:
        return None
    maximum = int(numba_config.NUMBA_NUM_THREADS)
    requested = int(n_jobs)
    if requested == -1:
        return maximum
    if requested > maximum:
        raise ValueError(f"n_jobs={requested} exceeds the available Numba thread maximum ({maximum}).")
    return requested


@contextmanager
def _numba_thread_scope(n_jobs: int | None):
    """Temporarily apply one Numba thread mask and always restore it."""
    resolved = _resolve_numba_thread_count(n_jobs)
    previous = get_num_threads()
    if resolved is not None and resolved != previous:
        set_num_threads(resolved)
    try:
        yield resolved if resolved is not None else previous
    finally:
        if get_num_threads() != previous:
            set_num_threads(previous)


def _run_target_comparisons(
    target_models: list[GenericModel],
    n_jobs: int | None,
    worker: Callable[[GenericModel], _ComparisonJobResult],
    *,
    progress: bool | None = False,
    progress_desc: str | None = None,
    progress_leave: bool = True,
) -> list[_ComparisonJobResult]:
    """Execute targets sequentially while numerical kernels use Numba threads."""
    if not target_models:
        return []

    del n_jobs
    return [
        worker(target_model)
        for target_model in iter_progress(
            target_models,
            enabled=progress,
            desc=progress_desc,
            total=len(target_models),
            leave=progress_leave,
        )
    ]


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
    with _numba_thread_scope(effective_config["n_jobs"]):
        return strategy_fn(model1, model2, sequences, effective_config)


def compare_one_to_many(
    query_model: GenericModel,
    target_models,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
    *,
    progress: bool | None = False,
    progress_desc: str | None = None,
    progress_leave: bool = True,
) -> list[ComparisonResult]:
    """Main entry point for one-vs-many motif comparison."""
    effective_config = replace(config, background=background) if background is not None else config

    with _numba_thread_scope(effective_config["n_jobs"]):
        if strategy == "profile":
            return _compare_profile_one_to_many(
                query_model,
                target_models,
                sequences,
                effective_config,
                progress=progress,
                progress_desc=progress_desc,
                progress_leave=progress_leave,
            )
        if strategy == "motif":
            return _compare_motif_one_to_many(
                query_model,
                target_models,
                sequences,
                effective_config,
                progress=progress,
                progress_desc=progress_desc,
                progress_leave=progress_leave,
            )
    available = ", ".join(sorted(registry))
    raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")
