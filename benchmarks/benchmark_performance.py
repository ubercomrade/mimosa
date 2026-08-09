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
import resource
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-count", type=int, default=2_000)
    parser.add_argument("--target-counts", default="1,64,128")
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-logerr", type=float, default=2.0)
    parser.add_argument("--memory-budget-bytes", type=int, default=1 << 30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _repeat_batch(batch, count):
    from mimosa import EncodedSequences

    return EncodedSequences.from_rows([batch[index % len(batch)] for index in range(count)])


def _models():
    from mimosa.io import read_model

    return [
        read_model(ROOT / "examples" / filename)
        for filename in ("foxa2.meme", "gata2.meme", "gata4.meme", "pif4.meme")
    ]


def _targets(models, count):
    from mimosa import PWM

    result = []
    for index in range(count):
        model = models[index % len(models)]
        result.append(PWM(f"{model.name}#{index}", model.weights, model.background))
    return result


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _cache_bytes(path):
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


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
    from mimosa import compare_many, prepare_profile
    from mimosa.cache import Cache

    set_num_threads(threads)
    cold_path = cache_path / "cold"
    shutil.rmtree(cold_path, ignore_errors=True)
    cold_cache = Cache(
        cold_path,
        memory_budget_bytes=args.memory_budget_bytes,
    )
    prepared_query = prepare_profile(
        models[0],
        sequences,
        background=background,
        min_logerr=args.min_logerr,
    )
    prepared_targets = [
        prepare_profile(
            target,
            sequences,
            background=background,
            min_logerr=args.min_logerr,
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
