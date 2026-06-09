#!/usr/bin/env python3
"""End-to-end benchmark for profile comparisons in a one-to-many setting.

This script measures the full `run_comparison()` loop for one query motif against
many target motifs. It can benchmark:
1. repeated passes with the same in-memory model objects (`reuse` scenario)
2. repeated passes with freshly constructed model objects (`fresh` scenario)

The `fresh` scenario is useful for estimating the benefit of on-disk profile cache.

Example:
    uv run python examples/benchmark_profile_one_to_many.py
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mimosa.api import compare_one_to_many
from mimosa.batches import SequenceBatch, make_sequence_batch
from mimosa.cache import clear_cache
from mimosa.comparison import create_comparator_config
from mimosa.models import GenericModel


def _build_sequences(num_sequences: int, seq_length: int, seed: int) -> SequenceBatch:
    rng = np.random.default_rng(seed)
    rows = [rng.integers(0, 4, size=seq_length, dtype=np.int8) for _ in range(num_sequences)]
    return make_sequence_batch(rows)


def _normalize_pwm_columns(pwm: np.ndarray) -> np.ndarray:
    centered = pwm - np.max(pwm, axis=0, keepdims=True)
    exp = np.exp(centered, dtype=np.float32)
    probs = exp / np.sum(exp, axis=0, keepdims=True)
    return np.log((probs + np.float32(1e-4)) / np.float32(0.25)).astype(np.float32)


def _prepare_model_bank(num_targets: int, motif_length: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    query_core = _normalize_pwm_columns(rng.normal(size=(4, motif_length)).astype(np.float32))
    target_cores: list[np.ndarray] = []

    for idx in range(num_targets):
        if idx % 3 == 0:
            target_core = (query_core + rng.normal(loc=0.0, scale=0.25, size=query_core.shape)).astype(np.float32)
        else:
            target_core = rng.normal(loc=0.0, scale=1.0, size=(4, motif_length)).astype(np.float32)
        target_cores.append(_normalize_pwm_columns(target_core))

    return query_core, target_cores


def _materialize_models(
    query_core: np.ndarray, target_cores: Sequence[np.ndarray]
) -> tuple[GenericModel, list[GenericModel]]:
    query = GenericModel(
        type_key="pwm",
        name="query",
        representation=np.vstack([query_core, np.min(query_core, axis=0, keepdims=True)]).astype(np.float32),
        length=query_core.shape[1],
        config={"kmer": 1},
    )

    targets = [
        GenericModel(
            type_key="pwm",
            name=f"target_{idx:04d}",
            representation=np.vstack([core, np.min(core, axis=0, keepdims=True)]).astype(np.float32),
            length=core.shape[1],
            config={"kmer": 1},
        )
        for idx, core in enumerate(target_cores)
    ]
    return query, targets


def _run_one_to_many(
    query: GenericModel,
    targets: Sequence[GenericModel],
    sequences: SequenceBatch,
    metric: str,
    min_logfpr: float,
    search_range: int,
    cache_mode: str,
    cache_dir: str,
) -> dict:
    comparator = create_comparator_config(
        metric=metric,
        min_logfpr=min_logfpr,
        search_range=search_range,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
    )

    start = time.perf_counter()
    results = compare_one_to_many(
        query=query,
        targets=list(targets),
        strategy="profile",
        sequences=sequences,
        comparator=comparator,
    )

    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "score_sum": float(sum(float(result["score"]) for result in results)),
        "offset_sum": int(sum(int(result["offset"]) for result in results)),
        "num_targets": len(results),
    }


def _warmup(
    sequences: SequenceBatch,
    metric: str,
    min_logfpr: float,
    search_range: int,
    cache_mode: str,
    cache_dir: str,
    query_core: np.ndarray,
    target_core: np.ndarray,
) -> None:
    query, targets = _materialize_models(query_core, [target_core])
    _run_one_to_many(query, targets, sequences, metric, min_logfpr, search_range, cache_mode, cache_dir)


def _parse_csv_values(raw_values: Iterable[str]) -> list[str]:
    values: list[str] = []
    for value in raw_values:
        for item in value.split(","):
            item = item.strip()
            if item:
                values.append(item)
    return values


def _parse_metrics(raw_values: Iterable[str]) -> list[str]:
    metrics = _parse_csv_values(raw_values)
    if not metrics:
        raise ValueError("At least one metric must be provided.")
    invalid = sorted(set(metrics) - {"co", "dice"})
    if invalid:
        raise ValueError(f"Unsupported metrics: {', '.join(invalid)}")
    return metrics


def _parse_thresholds(raw_values: Iterable[str]) -> list[float]:
    thresholds = [float(value) for value in _parse_csv_values(raw_values)]
    if not thresholds:
        raise ValueError("At least one min_logfpr value must be provided.")
    return thresholds


def _parse_cache_modes(raw_values: Iterable[str]) -> list[str]:
    modes = _parse_csv_values(raw_values)
    if not modes:
        raise ValueError("At least one cache mode must be provided.")
    invalid = sorted(set(modes) - {"off", "on"})
    if invalid:
        raise ValueError(f"Unsupported cache modes: {', '.join(invalid)}")
    return modes


def _parse_scenarios(raw_values: Iterable[str]) -> list[str]:
    scenarios = _parse_csv_values(raw_values)
    if not scenarios:
        raise ValueError("At least one scenario must be provided.")
    invalid = sorted(set(scenarios) - {"reuse", "fresh"})
    if invalid:
        raise ValueError(f"Unsupported scenarios: {', '.join(invalid)}")
    return scenarios


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark query-to-many profile comparisons.")
    parser.add_argument("--num-targets", type=int, default=1500, help="Number of target motifs. (default: %(default)s)")
    parser.add_argument(
        "--num-sequences", type=int, default=4000, help="Number of benchmark sequences. (default: %(default)s)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=100, help="Length of each benchmark sequence. (default: %(default)s)"
    )
    parser.add_argument("--motif-length", type=int, default=12, help="Motif length. (default: %(default)s)")
    parser.add_argument("--search-range", type=int, default=10, help="Alignment search range. (default: %(default)s)")
    parser.add_argument("--passes", type=int, default=2, help="Number of measured passes. (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed. (default: %(default)s)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["co"],
        help="Profile metrics to benchmark. Accepts comma-separated values. (default: co)",
    )
    parser.add_argument(
        "--min-logfpr",
        nargs="+",
        default=["0", "2", "4"],
        help="Thresholds to benchmark. Accepts comma-separated values. (default: 0 2 4)",
    )
    parser.add_argument(
        "--cache-modes",
        nargs="+",
        default=["off", "on"],
        help="Cache modes to benchmark. Accepts comma-separated values. (default: off on)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["reuse", "fresh"],
        help="Object-lifetime scenarios to benchmark. Accepts comma-separated values. (default: reuse fresh)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    metrics = _parse_metrics(args.metrics)
    thresholds = _parse_thresholds(args.min_logfpr)
    cache_modes = _parse_cache_modes(args.cache_modes)
    scenarios = _parse_scenarios(args.scenarios)

    sequences = _build_sequences(args.num_sequences, args.seq_length, args.seed)
    query_core, target_cores = _prepare_model_bank(args.num_targets, args.motif_length, args.seed)

    print("profile one-to-many benchmark")
    print(
        f"dataset: num_targets={args.num_targets}, num_sequences={args.num_sequences}, "
        f"seq_length={args.seq_length}, motif_length={args.motif_length}, search_range={args.search_range}"
    )
    print(f"passes={args.passes}, seed={args.seed}")
    print("")
    print("scenario\tcache\tmetric\tmin_logfpr\tpass\ttotal_s\tms_per_target\tscore_sum\toffset_sum")

    with tempfile.TemporaryDirectory(prefix="mimosa-bench-", dir="/tmp") as tmp_dir:
        cache_root = Path(tmp_dir) / "profile-cache"

        for metric in metrics:
            for threshold in thresholds:
                for cache_mode in cache_modes:
                    clear_cache(str(cache_root))
                    _warmup(
                        sequences=sequences,
                        metric=metric,
                        min_logfpr=threshold,
                        search_range=args.search_range,
                        cache_mode=cache_mode,
                        cache_dir=str(cache_root),
                        query_core=query_core,
                        target_core=target_cores[0],
                    )
                    clear_cache(str(cache_root))

                    for scenario in scenarios:
                        clear_cache(str(cache_root))
                        reused_models: tuple[GenericModel, list[GenericModel]] | None = None
                        total_times: list[float] = []

                        for pass_idx in range(1, args.passes + 1):
                            if scenario == "reuse":
                                if reused_models is None:
                                    reused_models = _materialize_models(query_core, target_cores)
                                query, targets = reused_models
                            else:
                                query, targets = _materialize_models(query_core, target_cores)

                            result = _run_one_to_many(
                                query=query,
                                targets=targets,
                                sequences=sequences,
                                metric=metric,
                                min_logfpr=threshold,
                                search_range=args.search_range,
                                cache_mode=cache_mode,
                                cache_dir=str(cache_root),
                            )
                            total_times.append(result["elapsed_s"])
                            ms_per_target = (result["elapsed_s"] * 1000.0) / max(result["num_targets"], 1)
                            print(
                                f"{scenario}\t{cache_mode}\t{metric}\t{threshold:.2f}\t{pass_idx}\t"
                                f"{result['elapsed_s']:.3f}\t{ms_per_target:.3f}\t"
                                f"{result['score_sum']:.6f}\t{result['offset_sum']}"
                            )

                        if len(total_times) > 1:
                            print(
                                f"# summary\t{scenario}\t{cache_mode}\t{metric}\t{threshold:.2f}\t"
                                f"mean={statistics.fmean(total_times):.3f}s\t"
                                f"best={min(total_times):.3f}s\tworst={max(total_times):.3f}s"
                            )


if __name__ == "__main__":
    main()
