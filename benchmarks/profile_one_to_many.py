"""Benchmark warm profile one-to-many throughput without input-generation time."""

from __future__ import annotations

import argparse
import hashlib
import platform
import resource
import sys
import time

import numba
import numpy as np
from numba import get_num_threads, set_num_threads

from mimosa.batches import make_random_sequence_batch
from mimosa.cache import clear_cache
from mimosa.comparison import profile as profile_strategy
from mimosa.comparison.config import create_comparator_config
from mimosa.comparison.runner import compare_one_to_many
from mimosa.handlers import pwm_model_from_pfm


def _make_models(count: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    return [
        pwm_model_from_pfm(rng.random((4, length), dtype=np.float32) + np.float32(0.01), f"target-{index}", length)
        for index in range(count)
    ]


def _checksum(results) -> str:
    payload = "|".join(
        f"{result.target}:{result.score:.9g}:{result.offset}:{result.orientation}:{result.n_sites}"
        for result in results
    )
    return hashlib.blake2b(payload.encode("ascii"), digest_size=16).hexdigest()


def _measure(run, query):
    started = time.perf_counter()
    phases: dict[str, float] = {}

    def timed(name, function):
        def wrapper(*args, **kwargs):
            phase_started = time.perf_counter()
            result = function(*args, **kwargs)
            phase_name = name
            if name == "preparation":
                phase_name = "query_preparation" if args[0] is query else "target_preparation"
            phases[phase_name] = phases.get(phase_name, 0.0) + time.perf_counter() - phase_started
            return result

        return wrapper

    originals = {
        "_prepare_profile_model": profile_strategy._prepare_profile_model,
        "_resolve_raw_profile_bundle": profile_strategy._resolve_raw_profile_bundle,
        "_apply_profile_normalizer": profile_strategy._apply_profile_normalizer,
        "_collect_anchor_sites": profile_strategy._collect_anchor_sites,
        "_score_profile_candidates": profile_strategy._score_profile_candidates,
    }
    profile_strategy._prepare_profile_model = timed("preparation", originals["_prepare_profile_model"])
    profile_strategy._resolve_raw_profile_bundle = timed("target_scan", originals["_resolve_raw_profile_bundle"])
    profile_strategy._apply_profile_normalizer = timed("normalization", originals["_apply_profile_normalizer"])
    profile_strategy._collect_anchor_sites = timed("anchor_preparation", originals["_collect_anchor_sites"])
    profile_strategy._score_profile_candidates = timed("alignment", originals["_score_profile_candidates"])
    try:
        results = run()
    finally:
        for name, function in originals.items():
            setattr(profile_strategy, name, function)
    phases["total"] = time.perf_counter() - started
    return phases, _checksum(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--sequence-count", type=int, default=10_000)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--motif-length", type=int, default=12)
    parser.add_argument("--cache-dir", help="Enable the profile disk cache in this directory.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear --cache-dir before measuring.")
    args = parser.parse_args()

    if args.clear_cache:
        if not args.cache_dir:
            parser.error("--clear-cache requires --cache-dir")
        clear_cache(args.cache_dir)

    set_num_threads(args.threads)
    sequences = make_random_sequence_batch(args.sequence_count, args.sequence_length, args.seed)
    background = make_random_sequence_batch(args.sequence_count, args.sequence_length, args.seed + 1)
    query = _make_models(1, args.motif_length, args.seed + 2)[0]
    targets = _make_models(args.target_count, args.motif_length, args.seed + 3)
    config = create_comparator_config(
        metric="co",
        search_range=5,
        window_radius=5,
        realign_window=3,
        min_logfpr=2.0,
        n_jobs=args.threads,
        cache_mode="on" if args.cache_dir else "off",
        cache_dir=args.cache_dir or ".",
    )

    def run():
        return compare_one_to_many(query, targets, "profile", config, sequences, background)

    cold_phases, checksum = _measure(run, query)
    first_warm_phases, first_checksum = _measure(run, query)
    warm_samples = [_measure(run, query) for _ in range(max(args.repeats - 1, 1))]
    checksums = {checksum, first_checksum, *(sample[1] for sample in warm_samples)}
    if len(checksums) != 1:
        raise RuntimeError(f"non-deterministic checksum: {sorted(checksums)}")

    values_bytes = int(sequences["values"].nbytes + background["values"].nbytes)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    warm_seconds = [sample[0]["total"] for sample in warm_samples]
    print("cold_s\tfirst_warm_s\twarm_median_s\tpeak_rss_kb\tbatch_bytes\tchecksum")
    print(
        f"{cold_phases['total']:.6f}\t{first_warm_phases['total']:.6f}\t{np.median(warm_seconds):.6f}\t{rss_kb}"
        f"\t{values_bytes}\t{checksum}"
    )
    print("stage\tfirst_warm_s\twarm_median_s")
    for stage in (
        "query_preparation",
        "target_preparation",
        "target_scan",
        "normalization",
        "anchor_preparation",
        "alignment",
        "total",
    ):
        samples = [sample[0].get(stage, 0.0) for sample in warm_samples]
        print(f"{stage}\t{first_warm_phases.get(stage, 0.0):.6f}\t{np.median(samples):.6f}")
    print(
        "environment"
        f" python={sys.version.split()[0]} numpy={np.__version__} numba={numba.__version__}"
        f" cpu={platform.processor() or platform.machine()} numba_threads={get_num_threads()}"
        f" threading_layer={numba.threading_layer()}"
    )
    print(
        "parameters"
        f" targets={args.target_count} sequences={args.sequence_count}x{args.sequence_length}"
        f" pwm_length={args.motif_length} metric=co search_range=5 window_radius=5 threshold=2.0"
        f" cache_mode={config['cache_mode']} cache_dir={args.cache_dir or 'none'}"
    )


if __name__ == "__main__":
    main()
