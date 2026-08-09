"""Measure the production compare_many path.

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
    parser.add_argument("--target-counts", default="1,64,128,256")
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
    from mimosa.cache import _make_preparation_context
    from mimosa.profiles.prepared import _prepare_profile

    phases = {
        "preparation_s": 0.0,
        "fit_normalize_anchors_s": 0.0,
        "cache_read_checksum_decode_s": 0.0,
    }
    for model in models:
        started = time.perf_counter()
        _prepare_profile(
            model,
            sequences,
            background=background,
            min_logerr=threshold,
            cache=cache,
        )
        phases["preparation_s"] += time.perf_counter() - started
    disk_cache = type(cache)(
        cache.directory,
        memory_budget_bytes=cache.memory_budget_bytes,
    )
    context = _make_preparation_context(sequences, background)
    started = time.perf_counter()
    for model in models:
        _prepare_profile(
            model,
            sequences,
            background=background,
            min_logerr=threshold,
            cache=disk_cache,
            _preparation_context=context,
        )
    phases["cache_read_checksum_decode_s"] = time.perf_counter() - started
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
    from mimosa.profiles.prepared import _prepare_profile

    set_num_threads(threads)
    cold_path = cache_path / "cold"
    shutil.rmtree(cold_path, ignore_errors=True)
    cold_cache = Cache(
        cold_path,
        memory_budget_bytes=args.memory_budget_bytes,
    )
    phase_cache = Cache(
        cache_path / "phase", memory_budget_bytes=args.memory_budget_bytes
    )
    phases = _preparation_phases(
        targets,
        sequences,
        background,
        phase_cache,
        args.min_logerr,
    )
    prepared_query = _prepare_profile(
        models[0],
        sequences,
        background=background,
        min_logerr=args.min_logerr,
        cache=phase_cache,
    )
    prepared_targets = [
        _prepare_profile(
            target,
            sequences,
            background=background,
            min_logerr=args.min_logerr,
            cache=phase_cache,
        )
        for target in targets
    ]
    compare_many(
        prepared_query,
        prepared_targets,
        min_logerr=args.min_logerr,
    )
    started = time.perf_counter()
    compare_many(
        prepared_query,
        prepared_targets,
        min_logerr=args.min_logerr,
    )
    prepared_alignment_s = time.perf_counter() - started
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

        timings.append(
            {
                "mode": mode,
                "wall_s": wall,
                "prepare_profiles_s": phases["preparation_s"] if mode == "cold" else 0.0,
                "cache_read_checksum_decode_s": (
                    phases["cache_read_checksum_decode_s"] if mode == "disk" else 0.0
                ),
                "normalization_anchors_s": phases["fit_normalize_anchors_s"] if mode == "cold" else 0.0,
                "prepared_alignment_s": prepared_alignment_s,
                "peak_rss_bytes": _rss_bytes(),
                "cache_bytes": _cache_bytes(cold_path),
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
                    "prepared_alignment_s",
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
                summary["samples"] = samples
                results.append(summary)
    payload = json.dumps({"benchmark": "mimosa-performance-v1", "results": results}, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
