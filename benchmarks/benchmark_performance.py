"""Measure the production compare_many pipeline.

Run from the repository root. The JSON output is machine-readable and reports
the total budget, per-worker Numba budget, and derived joblib worker count.
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
    parser.add_argument("--sequence-count", type=int, default=10_000)
    parser.add_argument("--target-counts", default="1,64,128,256")
    parser.add_argument("--total-threads", default="1,2,4,8")
    parser.add_argument("--numba-threads", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prepared-only", action="store_true")
    parser.add_argument("--min-logerr", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _positive_values(value, name):
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated positive integers") from exc
    if not values or any(item < 1 for item in values):
        raise SystemExit(f"{name} must be comma-separated positive integers")
    return values


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

    return [
        PWM(
            f"{models[index % len(models)].name}#{index}",
            models[index % len(models)].weights,
            models[index % len(models)].background,
        )
        for index in range(count)
    ]


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _cache_bytes(path):
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _compare_many(models, targets, sequences, background, cache, total_threads, inner_threads, threshold):
    from mimosa import compare_many

    return compare_many(
        models[0],
        targets,
        sequences,
        background=background,
        min_logerr=threshold,
        cache=cache,
        total_threads=total_threads,
        inner_threads=inner_threads,
    )


def _warmup(models, sequences, background, total_threads, inner_threads, threshold):
    from numba import set_num_threads

    set_num_threads(inner_threads)
    warm_sequences = _repeat_batch(sequences, min(64, len(sequences)))
    _compare_many(
        models,
        [models[1]] * 64,
        warm_sequences,
        background,
        None,
        total_threads,
        inner_threads,
        threshold,
    )


def _measure_full_pipeline(
    models,
    targets,
    sequences,
    background,
    cache_path,
    total_threads,
    inner_threads,
    threshold,
):
    from numba import set_num_threads
    from mimosa.cache import Cache, clearcache

    set_num_threads(inner_threads)
    shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir(parents=True)
    cold_cache = Cache(cache_path)
    started = time.perf_counter()
    _compare_many(
        models,
        targets,
        sequences,
        background,
        cold_cache,
        total_threads,
        inner_threads,
        threshold,
    )
    cold_wall = time.perf_counter() - started
    cold_bytes = _cache_bytes(cache_path)

    disk_cache = Cache(cache_path)
    started = time.perf_counter()
    _compare_many(
        models,
        targets,
        sequences,
        background,
        disk_cache,
        total_threads,
        inner_threads,
        threshold,
    )
    disk_wall = time.perf_counter() - started
    disk_bytes = _cache_bytes(cache_path)
    clearcache(Cache(cache_path))
    return [
        {"mode": "cold", "wall_s": cold_wall, "cache_bytes": cold_bytes, "peak_rss_bytes": _rss_bytes()},
        {"mode": "disk", "wall_s": disk_wall, "cache_bytes": disk_bytes, "peak_rss_bytes": _rss_bytes()},
    ]


def _measure_prepared(models, targets, sequences, background, total_threads, inner_threads, threshold):
    from numba import set_num_threads
    from mimosa import compare_many, prepare_profile

    set_num_threads(inner_threads)
    query = prepare_profile(models[0], sequences, background=background, min_logerr=threshold)
    prepared_targets = [
        prepare_profile(target, sequences, background=background, min_logerr=threshold)
        for target in targets
    ]
    compare_many(
        query,
        prepared_targets,
        total_threads=total_threads,
        inner_threads=inner_threads,
        min_logerr=threshold,
    )
    started = time.perf_counter()
    compare_many(
        query,
        prepared_targets,
        total_threads=total_threads,
        inner_threads=inner_threads,
        min_logerr=threshold,
    )
    return {
        "mode": "prepared",
        "wall_s": time.perf_counter() - started,
        "cache_bytes": 0,
        "peak_rss_bytes": _rss_bytes(),
    }


def main():
    args = _arguments()
    if args.sequence_count < 1 or args.repeats < 1:
        raise SystemExit("sequence-count and repeats must be positive")
    target_counts = _positive_values(args.target_counts, "target-counts")
    total_threads = _positive_values(args.total_threads, "total-threads")
    numba_threads = _positive_values(args.numba_threads, "numba-threads")
    if any(item > 4 for item in numba_threads):
        raise SystemExit("numba-threads values must be between 1 and 4")
    configurations = [
        (total, inner)
        for total in total_threads
        for inner in numba_threads
        if total % inner == 0
    ]
    if not configurations:
        raise SystemExit("no total/numba thread pair is divisible")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/mimosa-numba-cache")
    os.environ["NUMBA_NUM_THREADS"] = str(max(inner for _, inner in configurations))

    from mimosa.io.fasta import read_fasta

    foreground, _ = read_fasta(ROOT / "examples" / "foreground.fa")
    background, _ = read_fasta(ROOT / "examples" / "background.fa")
    sequences = _repeat_batch(foreground, args.sequence_count)
    models = _models()
    results = []
    modes = ("prepared",) if args.prepared_only else ("cold", "disk")
    with tempfile.TemporaryDirectory(prefix="mimosa-performance-") as temporary:
        root = Path(temporary)
        for total, inner in configurations:
            _warmup(models, sequences, background, total, inner, args.min_logerr)
            workers = total // inner
            for target_count in target_counts:
                targets = _targets(models, target_count)
                samples = []
                for _ in range(args.repeats):
                    if args.prepared_only:
                        samples.append(
                            _measure_prepared(
                                models,
                                targets,
                                sequences,
                                background,
                                total,
                                inner,
                                args.min_logerr,
                            )
                        )
                    else:
                        samples.extend(
                            _measure_full_pipeline(
                                models,
                                targets,
                                sequences,
                                background,
                                root / f"t{total}-n{inner}-targets{target_count}",
                                total,
                                inner,
                                args.min_logerr,
                            )
                        )
                summary = {
                    "total_threads": total,
                    "numba_threads": inner,
                    "joblib_workers": workers,
                    "target_count": target_count,
                    "samples": samples,
                }
                for mode in modes:
                    values = [sample["wall_s"] for sample in samples if sample["mode"] == mode]
                    summary[f"{mode}_wall_s_median"] = statistics.median(values)
                    summary[f"{mode}_targets_per_s"] = target_count / summary[f"{mode}_wall_s_median"]
                    for field in ("cache_bytes", "peak_rss_bytes"):
                        summary[f"{mode}_{field}_median"] = statistics.median(
                            sample[field] for sample in samples if sample["mode"] == mode
                        )
                results.append(summary)

    serial = {
        (result["target_count"], mode): result[f"{mode}_wall_s_median"]
        for result in results
        if result["total_threads"] == 1 and result["numba_threads"] == 1
        for mode in modes
    }
    for result in results:
        for mode in modes:
            baseline = serial.get((result["target_count"], mode))
            if baseline is not None:
                result[f"{mode}_speedup_vs_serial"] = baseline / result[f"{mode}_wall_s_median"]
    payload = json.dumps({"benchmark": "mimosa-performance-v3", "results": results}, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
