"""Measure the production compare_many path and its major phases.

Run from the repository root, for example::

    uv run python benchmarks/benchmark_performance.py --repeats 3

The JSON output is intentionally machine-readable.  Numba is warmed up before
the measured repetitions, and the reported summaries use the median repetition.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-count", type=int, default=10_000)
    parser.add_argument("--target-counts", default="64,128,256")
    parser.add_argument("--threads", default="1,2,4,6,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-logerr", type=float, default=2.0)
    parser.add_argument("--memory-budget-bytes", type=int, default=1 << 30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _repeat_batch(batch, count):
    from mimosa import EncodedSequences

    rows = [batch[index] for index in range(len(batch))]
    return EncodedSequences.from_rows(
        [rows[index % len(rows)] for index in range(count)]
    )


def _models():
    from mimosa.io.models import read_meme
    from mimosa.models import pwm_from_pfm

    result = []
    for filename in ("foxa2.meme", "gata2.meme", "gata4.meme", "pif4.meme"):
        name, pfm = read_meme(str(ROOT / "examples" / filename))
        result.append(pwm_from_pfm(pfm, name=name))
    return result


def _targets(models, count):
    from mimosa import PWM

    result = []
    for index in range(count):
        model = models[index % len(models)]
        result.append(PWM(f"{model.name}#{index}", model.weights, model.background))
    return result


def _rss_bytes():
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _cache_bytes(path):
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _preparation_phases(models, sequences, background, cache, threshold):
    from mimosa.cache import (
        _encode_prepared_profile,
        _make_preparation_context,
        _store_prepared_profile,
        _cached_prepared_profile,
        prepared_profile_cache_key,
    )
    from mimosa.profiles.anchors import collect_both_anchors
    from mimosa.profiles.normalization import HybridEmpiricalLogTail, _fit_normalize
    from mimosa.profiles.prepared import PreparedProfile
    from mimosa.compare import _TARGET_BATCH_SIZE
    from mimosa.scan import _scan_models_batch

    normalization = HybridEmpiricalLogTail()
    context = _make_preparation_context(sequences, background)
    phases = {
        "scan_models_batch_s": 0.0,
        "fit_normalize_anchors_s": 0.0,
        "encode_s": 0.0,
        "cache_set_s": 0.0,
        "cache_read_checksum_decode_s": 0.0,
    }

    for batch_start in range(0, len(models), _TARGET_BATCH_SIZE):
        model_batch = models[batch_start : batch_start + _TARGET_BATCH_SIZE]
        started = time.perf_counter()
        raw = _scan_models_batch(model_batch, sequences)
        background_raw = None
        if background is not None and background is not sequences:
            background_raw = _scan_models_batch(model_batch, background)
        phases["scan_models_batch_s"] += time.perf_counter() - started

        for local_index, model in enumerate(model_batch):
            started = time.perf_counter()
            raw_pair = raw.pair(local_index)
            if background_raw is None:
                _, normalized = _fit_normalize(
                    normalization, raw_pair, tail_logerr=threshold
                )
            else:
                _, normalized = _fit_normalize(
                    normalization,
                    raw_pair,
                    calibration=background_raw.pair(local_index),
                    tail_logerr=threshold,
                )
            anchors = collect_both_anchors(normalized, threshold)
            phases["fit_normalize_anchors_s"] += time.perf_counter() - started
            profile = PreparedProfile._from_validated(
                model.name, normalized, anchors, threshold, normalization
            )
            started = time.perf_counter()
            _encode_prepared_profile(profile)
            phases["encode_s"] += time.perf_counter() - started
            key = prepared_profile_cache_key(
                cache,
                model,
                sequences,
                background=background,
                min_logerr=threshold,
                normalization=normalization,
            )
            started = time.perf_counter()
            _store_prepared_profile(cache, key, profile)
            phases["cache_set_s"] += time.perf_counter() - started
    disk_cache = type(cache)(
        cache.directory,
        memory_budget_bytes=cache.memory_budget_bytes,
    )
    started = time.perf_counter()
    for model in models:
        _cached_prepared_profile(
            disk_cache,
            model,
            sequences,
            background,
            threshold,
            normalization,
            context,
        )
    phases["cache_read_checksum_decode_s"] = time.perf_counter() - started
    phases["preparation_s"] = sum(
        phases[name]
        for name in (
            "scan_models_batch_s",
            "fit_normalize_anchors_s",
            "encode_s",
            "cache_set_s",
        )
    )
    return phases


def _warmup(models, sequences, threads, threshold):
    from numba import set_num_threads
    from mimosa import compare_many, prepare_profile

    set_num_threads(threads)
    warm_sequences = _repeat_batch(sequences, min(64, len(sequences)))
    query = prepare_profile(models[0], warm_sequences, min_logerr=threshold)
    compare_many(
        query,
        [models[1]] * 64,
        warm_sequences,
        min_logerr=threshold,
    )


def _measure_modes(models, targets, sequences, background, cache_path, args, threads):
    from numba import set_num_threads
    from mimosa import compare_many
    from mimosa.cache import Cache
    from mimosa.compare import (
        _TARGET_BATCH_SIZE,
        _compare_many_prepared_parallel,
        _prepare_side,
    )
    from mimosa.parallel import MIN_PARALLEL_TARGETS, scan_dispatch_path
    from mimosa.scan import _scan_offsets
    from mimosa.profiles.alignment import ProfileConfig

    set_num_threads(threads)
    cold_path = cache_path / "cold"
    shutil.rmtree(cold_path, ignore_errors=True)
    cold_cache = Cache(
        cold_path,
        memory_budget_bytes=args.memory_budget_bytes,
    )
    phases = _preparation_phases(
        targets,
        sequences,
        background,
        Cache(cache_path / "phase", memory_budget_bytes=args.memory_budget_bytes),
        args.min_logerr,
    )
    groups = {}
    representatives = {}
    for target in targets:
        key = (type(target).__name__, target.motif_length)
        groups[key] = groups.get(key, 0) + 1
        representatives.setdefault(key, target)
    scan_paths = [
        scan_dispatch_path(
            int(_scan_offsets(sequences, representatives[key])[-1]),
            rows=len(sequences),
            groups=size,
        )
        for key, size in groups.items()
    ]
    scan_path = (
        "model-parallel"
        if "model-parallel" in scan_paths
        else "row-parallel"
        if "row-parallel" in scan_paths
        else "serial"
    )

    timings = []
    for mode in ("cold", "disk", "memory"):
        if mode == "cold":
            cache = cold_cache
        elif mode == "disk":
            cache = Cache(
                cold_path,
                memory_budget_bytes=args.memory_budget_bytes,
            )
        else:
            cache = cold_cache
        started = time.perf_counter()
        compare_many(
            models[0],
            targets,
            sequences,
            background=background,
            min_logerr=args.min_logerr,
            cache=cache,
        )
        wall = time.perf_counter() - started

        if (
            len(targets) >= MIN_PARALLEL_TARGETS
            and _TARGET_BATCH_SIZE >= MIN_PARALLEL_TARGETS
        ):
            phase_cache = Cache(
                cache.directory,
                memory_budget_bytes=args.memory_budget_bytes,
            )
            query = _prepare_side(
                models[0],
                sequences,
                background,
                args.min_logerr,
                None,
                phase_cache,
            )
            prepared_targets = [
                _prepare_side(
                    target,
                    sequences,
                    background,
                    args.min_logerr,
                    None,
                    phase_cache,
                )
                for target in targets
            ]
            parallel_phases = {}
            _compare_many_prepared_parallel(
                query,
                prepared_targets,
                ProfileConfig(min_logerr=args.min_logerr),
                phase_cache,
                None,
                parallel_phases,
            )
        else:
            parallel_phases = {}

        timings.append(
            {
                "mode": mode,
                "wall_s": wall,
                "prepare_profiles_s": phases["preparation_s"] if mode == "cold" else 0.0,
                "cache_read_checksum_decode_s": (
                    phases["cache_read_checksum_decode_s"] if mode == "disk" else 0.0
                ),
                "normalization_anchors_s": phases["fit_normalize_anchors_s"] if mode == "cold" else 0.0,
                "packing_s": parallel_phases.get("packing", 0.0),
                "alignment_kernel_s": parallel_phases.get("alignment_kernel", 0.0),
                "peak_rss_bytes": _rss_bytes(),
                "cache_bytes": _cache_bytes(cold_path),
                "dispatch_path": (
                    "target-parallel"
                    if len(targets) >= MIN_PARALLEL_TARGETS
                    and _TARGET_BATCH_SIZE >= MIN_PARALLEL_TARGETS
                    else "serial"
                ),
                "scan_dispatch_path": scan_path,
            }
        )
    return timings


def main():
    args = _arguments()
    if args.sequence_count < 1 or args.repeats < 1:
        raise SystemExit("sequence-count and repeats must be positive")
    target_counts = [int(value) for value in args.target_counts.split(",") if value]
    threads = [int(value) for value in args.threads.split(",") if value]
    if not target_counts or not threads or any(value < 1 for value in (*target_counts, *threads)):
        raise SystemExit("target-counts and threads must contain positive integers")
    os.environ["NUMBA_NUM_THREADS"] = str(max(threads))

    from mimosa.io.fasta import read_fasta

    foreground, _ = read_fasta(str(ROOT / "examples" / "foreground.fa"))
    background, _ = read_fasta(str(ROOT / "examples" / "background.fa"))
    sequences = _repeat_batch(foreground, args.sequence_count)
    models = _models()
    results = []
    with tempfile.TemporaryDirectory(prefix="mimosa-performance-") as temporary:
        root = Path(temporary)
        for threads_value in threads:
            _warmup(models, sequences, threads_value, args.min_logerr)
            for target_count in target_counts:
                targets = _targets(models, target_count)
                samples = []
                for _ in range(args.repeats):
                    samples.extend(
                        _measure_modes(
                            models,
                            targets,
                            sequences,
                            background,
                            root / f"t{threads_value}-n{target_count}",
                            args,
                            threads_value,
                        )
                    )
                summary = {"threads": threads_value, "target_count": target_count}
                for field in (
                    "wall_s",
                    "prepare_profiles_s",
                    "cache_read_checksum_decode_s",
                    "normalization_anchors_s",
                    "packing_s",
                    "alignment_kernel_s",
                    "peak_rss_bytes",
                    "cache_bytes",
                ):
                    for mode in ("cold", "disk", "memory"):
                        values = [
                            sample[field]
                            for sample in samples
                            if sample["mode"] == mode
                        ]
                        summary[f"{mode}_{field}_median"] = statistics.median(values)
                summary["dispatch_path"] = samples[0]["dispatch_path"]
                summary["scan_dispatch_path"] = samples[0]["scan_dispatch_path"]
                summary["samples"] = samples
                results.append(summary)
    payload = json.dumps({"benchmark": "mimosa-performance-v1", "results": results}, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
