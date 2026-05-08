#!/usr/bin/env python3
"""Microbenchmark for the profile scorer kernels.

Examples:
    uv run python examples/benchmark_fast_profile_score.py
    uv run python examples/benchmark_fast_profile_score.py --threads 1 4 8 --modes pairwise orientations
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
from numba import get_num_threads, set_num_threads

from mimosa.batches import make_score_batch, make_strand_bundle
from mimosa.comparison import PROFILE_ORIENTATION_PAIRS, _score_profile_candidates, _select_best_orientation, create_comparator_config
from mimosa.functions import prepare_profile_bundle


def _generate_score_rows(num_rows: int, row_length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for _ in range(num_rows):
        rows.append(rng.gamma(shape=1.8, scale=1.0, size=row_length).astype(np.float32))
    return rows


def _generate_pairwise_profiles(num_rows: int, row_length: int, seed: int):
    """Build one pair of dense score batches."""
    rows1 = _generate_score_rows(num_rows, row_length, seed)
    rows2 = _generate_score_rows(num_rows, row_length, seed + 1)
    return make_score_batch(rows1), make_score_batch(rows2)


def _generate_orientation_bundles(num_rows: int, row_length: int, seed: int):
    """Build one pair of two-strand profile bundles."""
    query_plus, query_minus = (
        make_score_batch(_generate_score_rows(num_rows, row_length, seed)),
        make_score_batch(_generate_score_rows(num_rows, row_length, seed + 1)),
    )
    target_plus, target_minus = (
        make_score_batch(_generate_score_rows(num_rows, row_length, seed + 2)),
        make_score_batch(_generate_score_rows(num_rows, row_length, seed + 3)),
    )
    return (
        make_strand_bundle(query_plus, query_minus),
        make_strand_bundle(target_plus, target_minus),
    )


@contextmanager
def _thread_scope(num_threads: int):
    """Temporarily apply one Numba thread count."""
    previous = get_num_threads()
    set_num_threads(num_threads)
    try:
        yield
    finally:
        set_num_threads(previous)


def _parse_metric_list(raw_values: Iterable[str]) -> list[str]:
    metrics = []
    for value in raw_values:
        for item in value.split(","):
            item = item.strip()
            if item:
                metrics.append(item)
    if not metrics:
        raise ValueError("At least one metric must be provided.")
    valid_metrics = {"co", "co_rowwise", "dice", "dice_rowwise", "cosine"}
    invalid = sorted(set(metrics) - valid_metrics)
    if invalid:
        raise ValueError(f"Unsupported metrics: {', '.join(invalid)}")
    return metrics


def _parse_threshold_list(raw_values: Iterable[str]) -> list[float]:
    thresholds: list[float] = []
    for value in raw_values:
        for item in value.split(","):
            item = item.strip()
            if item:
                thresholds.append(float(item))
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return thresholds


def _parse_mode_list(raw_values: Iterable[str]) -> list[str]:
    modes = []
    for value in raw_values:
        for item in value.split(","):
            item = item.strip()
            if item:
                modes.append(item)
    if not modes:
        raise ValueError("At least one benchmark mode must be provided.")
    invalid = sorted(set(modes) - {"pairwise", "orientations"})
    if invalid:
        raise ValueError(f"Unsupported modes: {', '.join(invalid)}")
    return modes


def _time_call(func, warmups: int, repeats: int) -> tuple[list[float], object]:
    """Measure one callable after JIT warmup."""
    last_result = None
    for _ in range(warmups):
        last_result = func()

    durations_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func()
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    return durations_ms, last_result


def _benchmark_pairwise(
    profile1,
    profile2,
    comparator_config: dict,
    num_threads: int,
    warmups: int,
    repeats: int,
) -> dict:
    with _thread_scope(num_threads):
        durations_ms, last_result = _time_call(
            lambda: _select_best_orientation(
                _score_profile_candidates(profile1, profile2, [("++", 0, 0)], comparator_config)
            ),
            warmups,
            repeats,
        )

    score = float(last_result["score"])
    offset = int(last_result["shift"])
    total_positions = int(np.sum(profile1["lengths"]))
    mean_ms = statistics.fmean(durations_ms)
    return {
        "mode": "pairwise",
        "threads": num_threads,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(durations_ms),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "mpos_per_sec": (total_positions / (mean_ms / 1000.0)) / 1_000_000.0,
        "score": float(score),
        "offset": int(offset),
    }


def _benchmark_orientations(
    left_bundle,
    right_bundle,
    comparator_config: dict,
    num_threads: int,
    warmups: int,
    repeats: int,
) -> dict:
    with _thread_scope(num_threads):
        durations_ms, last_result = _time_call(
            lambda: _select_best_orientation(
                _score_profile_candidates(left_bundle, right_bundle, PROFILE_ORIENTATION_PAIRS, comparator_config)
            ),
            warmups,
            repeats,
        )

    total_positions = int(np.sum(left_bundle["lengths"])) * len(PROFILE_ORIENTATION_PAIRS)
    mean_ms = statistics.fmean(durations_ms)
    return {
        "mode": "orientations",
        "threads": num_threads,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(durations_ms),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "mpos_per_sec": (total_positions / (mean_ms / 1000.0)) / 1_000_000.0,
        "score": float(last_result["score"]),
        "offset": int(last_result["shift"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Microbenchmark fast_profile_score kernels on synthetic profile data.")
    parser.add_argument("--num-rows", type=int, default=4000, help="Number of profile rows. (default: %(default)s)")
    parser.add_argument("--row-length", type=int, default=100, help="Profile row width. (default: %(default)s)")
    parser.add_argument("--search-range", type=int, default=10, help="Alignment search range. (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=10, help="Timed repetitions per case. (default: %(default)s)")
    parser.add_argument("--warmups", type=int, default=3, help="Warm-up runs per case. (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed. (default: %(default)s)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["co", "dice"],
        help="Metrics to benchmark. Accepts comma-separated values. (default: co dice)",
    )
    parser.add_argument(
        "--min-logfpr",
        nargs="+",
        default=["0", "2", "4"],
        help="Thresholds to benchmark. Accepts comma-separated values. (default: 0 2 4)",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Numba thread counts to benchmark. (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["pairwise", "orientations"],
        help="Benchmark modes. Accepts comma-separated values. (default: pairwise orientations)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    metrics = _parse_metric_list(args.metrics)
    thresholds = _parse_threshold_list(args.min_logfpr)
    modes = _parse_mode_list(args.modes)
    pairwise_left_raw, pairwise_right_raw = _generate_pairwise_profiles(args.num_rows, args.row_length, args.seed)
    pairwise_left = prepare_profile_bundle(make_strand_bundle(pairwise_left_raw, pairwise_left_raw))
    pairwise_right = prepare_profile_bundle(make_strand_bundle(pairwise_right_raw, pairwise_right_raw))
    orientation_left_raw, orientation_right_raw = _generate_orientation_bundles(args.num_rows, args.row_length, args.seed)
    orientation_left = prepare_profile_bundle(orientation_left_raw)
    orientation_right = prepare_profile_bundle(orientation_right_raw)

    print("fast_profile_score microbenchmark")
    print(
        f"dataset: num_rows={args.num_rows}, row_length={args.row_length}, "
        f"total_positions={int(np.sum(pairwise_left['lengths']))}, search_range={args.search_range}"
    )
    print(f"repeats={args.repeats}, warmups={args.warmups}, seed={args.seed}")
    print("")
    print("mode\tmetric\tmin_logfpr\tthreads\tmean_ms\tmedian_ms\tmin_ms\tmax_ms\tMpos/s\tscore\toffset")

    for mode in modes:
        for metric in metrics:
            for threshold in thresholds:
                comparator_config = create_comparator_config(
                    search_range=args.search_range,
                    metric=metric,
                    min_logfpr=threshold,
                    n_permutations=0,
                )
                for num_threads in args.threads:
                    if mode == "pairwise":
                        result = _benchmark_pairwise(
                            pairwise_left,
                            pairwise_right,
                            comparator_config,
                            num_threads,
                            args.warmups,
                            args.repeats,
                        )
                    else:
                        result = _benchmark_orientations(
                            orientation_left,
                            orientation_right,
                            comparator_config,
                            num_threads,
                            args.warmups,
                            args.repeats,
                        )
                    print(
                        f"{result['mode']}\t"
                        f"{metric}\t"
                        f"{threshold:.2f}\t"
                        f"{result['threads']}\t"
                        f"{result['mean_ms']:.3f}\t"
                        f"{result['median_ms']:.3f}\t"
                        f"{result['min_ms']:.3f}\t"
                        f"{result['max_ms']:.3f}\t"
                        f"{result['mpos_per_sec']:.3f}\t"
                        f"{result['score']:.6f}\t"
                        f"{result['offset']}"
                    )


if __name__ == "__main__":
    main()
