"""Measure the production compare_many pipeline.

Run from the repository root. The JSON output is machine-readable and reports
the total budget, per-worker Numba budget, and derived joblib worker count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows does not provide resource.
    resource = None

ROOT = Path(__file__).resolve().parents[1]


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-count", type=int, default=10_000)
    parser.add_argument("--target-counts", default="1,64,128,256")
    parser.add_argument("--total-threads", default="1,2,4,8")
    parser.add_argument("--numba-threads", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prepared-only", action="store_true")
    parser.add_argument(
        "--allow-resource-overcommit",
        action="store_true",
        help="run configurations even when estimated temporary cache storage is unavailable",
    )
    parser.add_argument(
        "--cold-jit-cli",
        action="store_true",
        help="also measure one clean-process CLI invocation with an empty Numba cache",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="include BaMM and SiteGA targets and use a larger background batch",
    )
    parser.add_argument(
        "--skewed-rows",
        action="store_true",
        help="vary generated row lengths to exercise ragged scan and alignment paths",
    )
    parser.add_argument(
        "--thresholds",
        help="comma-separated min-logerr values; for example 0,1e-6,1,2,4",
    )
    parser.add_argument(
        "--phase-timings",
        action="store_true",
        help="add isolated scan, normalization, anchor, cache, alignment, and JSON timings",
    )
    parser.add_argument(
        "--worker-mode",
        choices=(
            "cache_miss_hot_jit",
            "cache_hit_hot_filesystem",
            "prepared",
            "cold_jit_cli",
        ),
        help=argparse.SUPPRESS,
    )
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


def _threshold_values(value, default):
    if value is None:
        return [default]
    try:
        values = [float(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise SystemExit("thresholds must be comma-separated finite numbers") from exc
    if not values or any(not math.isfinite(item) for item in values):
        raise SystemExit("thresholds must be comma-separated finite numbers")
    return values


def _repeat_batch(batch, count):
    from mimosa import EncodedSequences

    return EncodedSequences.from_rows([batch[index % len(batch)] for index in range(count)])


def _models(*, coverage=False):
    from mimosa.io import read_model

    models = [
        read_model(ROOT / "examples" / filename)
        for filename in ("foxa2.meme", "gata2.meme", "gata4.meme", "pif4.meme")
    ]
    if coverage:
        models.extend(
            (
                read_model(ROOT / "examples" / "myog.ihbcp"),
                read_model(ROOT / "examples" / "sitega_stat6.mat"),
            )
        )
    return models


def _targets(models, count, *, coverage=False):
    from mimosa import PWM

    pwm_targets = [
        PWM(
            f"{models[index % 4].name}#{index}",
            models[index % 4].weights,
            models[index % 4].background,
        )
        for index in range(count)
    ]
    if not coverage:
        return pwm_targets
    return ([models[-2], models[-1]] + pwm_targets)[:count]


def _skew_batch(batch, count):
    from mimosa import EncodedSequences

    rows = []
    for index in range(count):
        source = batch[index % len(batch)]
        ratio = 1 + index % 5
        rows.append(source[: max(1, source.size * ratio // 5)])
    return EncodedSequences.from_rows(rows)


def _parent_lifetime_max_rss_bytes():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _with_parent_rss(sample):
    rss = _parent_lifetime_max_rss_bytes()
    if rss is not None:
        sample["parent_lifetime_max_rss_bytes"] = rss
    return sample


def _cache_bytes(path):
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _estimated_cache_bytes_for_targets(one_target_cache_bytes, target_count):
    """Estimate query-plus-target cache space from a same-workload 1-target run."""
    return math.ceil(one_target_cache_bytes * (target_count + 1) / 2)


def _resource_limited_samples(modes, *, reason, required_bytes, available_bytes):
    return [
        {
            "mode": mode,
            "status": "resource_limited",
            "reason": reason,
            "required_bytes": required_bytes,
            "available_bytes": available_bytes,
        }
        for mode in modes
    ]


def _benchmark_payload(results):
    return {
        "benchmark": "mimosa-performance-v5",
        "rss_scope": "aggregate process-tree peak on Linux; parent fallback elsewhere",
        "results": results,
    }


def _write_benchmark_output(path, results):
    if path is None:
        return
    payload = json.dumps(_benchmark_payload(results), indent=2)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _measure_cache_miss_hot_jit(
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

    clearcache(Cache(cache_path))
    return _with_parent_rss(
        {
            "mode": "cache_miss_hot_jit",
            "wall_s": cold_wall,
            "cache_bytes": cold_bytes,
        }
    )


def _measure_cache_hit_hot_filesystem(
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
    _compare_many(
        models,
        targets,
        sequences,
        background,
        Cache(cache_path),
        total_threads,
        inner_threads,
        threshold,
    )
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
    return _with_parent_rss(
        {
            "mode": "cache_hit_hot_filesystem",
            "wall_s": disk_wall,
            "cache_bytes": disk_bytes,
        }
    )


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
    return _with_parent_rss(
        {
            "mode": "prepared",
            "wall_s": time.perf_counter() - started,
            "cache_bytes": 0,
        }
    )


def _phase_timings(models, targets, sequences, background, total_threads, inner_threads, threshold, cache_path):
    """Time isolated production primitives without changing the normal pipeline."""
    from mimosa import compare_many, prepare_profile, scan
    from mimosa.cache import Cache, _cached_mmap_prepared_profile, _store_normalized_profile, prepared_profile_cache_key
    from mimosa.io.bundles import model_fingerprint, sequence_fingerprint
    from mimosa.profiles.anchors import collect_both_anchors
    from mimosa.profiles.normalization import HybridEmpiricalLogTail, _fit_exact, normalize_bundle
    from mimosa.parallel import alignment_scratch_bytes

    timings = {}

    started = time.perf_counter()
    serialized_targets = pickle.dumps(targets, protocol=pickle.HIGHEST_PROTOCOL)
    timings["target_pipeline_pickle"] = time.perf_counter() - started
    timings["target_pipeline_pickle_bytes"] = len(serialized_targets)
    timings["target_identity_duplicate_count"] = len(targets) - len(
        {id(target) for target in targets}
    )

    started = time.perf_counter()
    raw = scan(models[0], sequences, strands="both")
    timings["scan"] = time.perf_counter() - started

    started = time.perf_counter()
    table = _fit_exact(HybridEmpiricalLogTail(), raw)
    normalized = normalize_bundle(table, raw)
    timings["normalize"] = time.perf_counter() - started

    started = time.perf_counter()
    collect_both_anchors(normalized, threshold, position_offset=models[0].left_context)
    timings["anchors"] = time.perf_counter() - started

    started = time.perf_counter()
    model_fingerprint(models[0])
    sequence_fingerprint(sequences)
    if background is not None:
        sequence_fingerprint(background)
    timings["fingerprint"] = time.perf_counter() - started

    prepared_query = prepare_profile(
        models[0], sequences, background=background, min_logerr=threshold
    )
    key = prepared_profile_cache_key(
        models[0], sequences, background=background, min_logerr=threshold
    )
    cache = Cache(cache_path, timings=timings)
    _store_normalized_profile(
        cache,
        key,
        prepared_query.name,
        prepared_query.bundle,
        prepared_query.normalization,
        prepared_query.site_start_offset,
    )

    hit_timings = {}
    if _cached_mmap_prepared_profile(Cache(cache_path, timings=hit_timings), key) is None:
        raise RuntimeError("phase cache read unexpectedly missed")
    timings.update(hit_timings)

    prepared_targets = [
        prepare_profile(target, sequences, background=background, min_logerr=threshold)
        for target in targets
    ]
    timings["alignment_parallel_scratch_bytes"] = alignment_scratch_bytes(
        len(prepared_query.bundle.forward), 2 * 10 + 1
    )
    started = time.perf_counter()
    results = compare_many(
        prepared_query,
        prepared_targets,
        total_threads=total_threads,
        inner_threads=inner_threads,
        min_logerr=threshold,
    )
    timings["alignment"] = time.perf_counter() - started

    started = time.perf_counter()
    json.dumps([result.to_dict() for result in results], sort_keys=True)
    timings["serialization"] = time.perf_counter() - started
    return timings


def _measure_cold_jit_cli(total_threads, inner_threads, threshold, root):
    """Measure the executable in a clean Python process and Numba cache."""
    cache_path = root / "cache"
    environment = dict(os.environ)
    environment["NUMBA_CACHE_DIR"] = str(root / "numba")
    environment["NUMBA_NUM_THREADS"] = str(inner_threads)
    command = [
        sys.executable,
        "-m",
        "mimosa.cli",
        "compare",
        "examples/foxa2.meme",
        "examples/gata2.meme",
        "--query-type",
        "pwm",
        "--target-type",
        "pwm",
        "--fasta",
        "examples/foreground.fa",
        "--min-logerr",
        str(threshold),
        "--numba-threads",
        str(inner_threads),
        "--cache-dir",
        str(cache_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=ROOT, env=environment
    )
    wall_s = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"cold CLI failed: {completed.stderr.strip()}")
    json.loads(completed.stdout)
    return {
        "mode": "cold_jit_cli",
        "wall_s": wall_s,
        "cache_bytes": _cache_bytes(cache_path),
    }


def _configurations(args):
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
    return target_counts, configurations


def _configure_numba_environment(configurations):
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "mimosa-numba-cache")
    )
    os.environ["NUMBA_NUM_THREADS"] = str(max(inner for _, inner in configurations))


def _worker_sample(args):
    target_counts, configurations = _configurations(args)
    if len(target_counts) != 1 or len(configurations) != 1:
        raise SystemExit("worker mode requires exactly one target and thread configuration")
    _configure_numba_environment(configurations)

    if args.worker_mode == "cold_jit_cli":
        total, inner = configurations[0]
        with tempfile.TemporaryDirectory(prefix="mimosa-performance-") as temporary:
            sample = _measure_cold_jit_cli(total, inner, args.min_logerr, Path(temporary))
        print(json.dumps(sample, sort_keys=True))
        return

    from mimosa.io.fasta import read_fasta

    foreground, _ = read_fasta(ROOT / "examples" / "foreground.fa")
    background, _ = read_fasta(ROOT / "examples" / "background.fa")
    sequences = (
        _skew_batch(foreground, args.sequence_count)
        if args.skewed_rows
        else _repeat_batch(foreground, args.sequence_count)
    )
    if args.coverage:
        background = _repeat_batch(background, max(len(background), 2 * args.sequence_count))
    models = _models(coverage=args.coverage)
    total, inner = configurations[0]
    targets = _targets(models, target_counts[0], coverage=args.coverage)
    _warmup(models, sequences, background, total, inner, args.min_logerr)
    with tempfile.TemporaryDirectory(prefix="mimosa-performance-") as temporary:
        root = Path(temporary)
        if args.worker_mode == "prepared":
            sample = _measure_prepared(
                models, targets, sequences, background, total, inner, args.min_logerr
            )
        elif args.worker_mode == "cache_miss_hot_jit":
            sample = _measure_cache_miss_hot_jit(
                models,
                targets,
                sequences,
                background,
                root / "cache",
                total,
                inner,
                args.min_logerr,
            )
        else:
            sample = _measure_cache_hit_hot_filesystem(
                models,
                targets,
                sequences,
                background,
                root / "cache",
                total,
                inner,
                args.min_logerr,
            )
        if args.phase_timings:
            sample["phase_wall_s"] = _phase_timings(
                models,
                targets,
                sequences,
                background,
                total,
                inner,
                args.min_logerr,
                root / "phase-cache",
            )
        sample["target_identity_duplicate_count"] = len(targets) - len(
            {id(target) for target in targets}
        )
    print(json.dumps(sample, sort_keys=True))


def _process_tree_rss_bytes(root_pid):
    pending = [root_pid]
    seen = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text()
            pending.extend(map(int, children.split()))
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


def _run_sample_subprocess(command):
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        return subprocess.run(command, capture_output=True, text=True), None
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    peak = 0
    while process.poll() is None:
        peak = max(peak, _process_tree_rss_bytes(process.pid))
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        peak,
    )


def _sample_in_subprocess(args, total, inner, target_count, threshold, mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--sequence-count",
        str(args.sequence_count),
        "--target-counts",
        str(target_count),
        "--total-threads",
        str(total),
        "--numba-threads",
        str(inner),
        "--repeats",
        "1",
        "--min-logerr",
        str(threshold),
    ]
    if args.coverage:
        command.append("--coverage")
    if args.skewed_rows:
        command.append("--skewed-rows")
    if args.phase_timings:
        command.append("--phase-timings")
    completed, aggregate_peak_rss = _run_sample_subprocess(command)
    if completed.returncode:
        # A high-cardinality worker can be killed by the OS before Python can
        # report MemoryError. Preserve the rest of the benchmark rather than
        # discarding all completed configurations.
        if completed.returncode < 0 or completed.returncode in {137, 143}:
            return {
                "mode": mode,
                "status": "resource_limited",
                "returncode": completed.returncode,
            }
        raise RuntimeError(
            f"benchmark worker {mode} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        sample = json.loads(completed.stdout)
        if aggregate_peak_rss is not None:
            sample["aggregate_peak_rss_bytes"] = aggregate_peak_rss
        return sample
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark worker {mode} emitted invalid JSON.") from exc


def main():
    args = _arguments()
    target_counts, configurations = _configurations(args)
    if args.worker_mode is not None:
        _worker_sample(args)
        return
    modes = (
        ("prepared",)
        if args.prepared_only
        else ("cache_miss_hot_jit", "cache_hit_hot_filesystem")
    )
    if args.cold_jit_cli and not args.prepared_only:
        modes = (*modes, "cold_jit_cli")
    thresholds = _threshold_values(args.thresholds, args.min_logerr)
    results = []
    for total, inner in configurations:
        for target_count in target_counts:
            for threshold in thresholds:
                active_modes = tuple(
                    mode
                    for mode in modes
                    if mode != "cold_jit_cli" or target_count == 1
                )
                one_target = next(
                    (
                        result
                        for result in results
                        if result["target_count"] == 1
                        and result["min_logerr"] == threshold
                        and result["input_suite"]
                        == ("heterogeneous" if args.coverage else "pwm")
                        and result["skewed_rows"] == args.skewed_rows
                        and "cache_miss_hot_jit_cache_bytes_median" in result
                    ),
                    None,
                )
                required_bytes = (
                    0
                    if one_target is None
                    else _estimated_cache_bytes_for_targets(
                        one_target["cache_miss_hot_jit_cache_bytes_median"],
                        target_count,
                    )
                )
                available_bytes = shutil.disk_usage(tempfile.gettempdir()).free
                if (
                    required_bytes
                    and required_bytes > available_bytes * 0.8
                    and not args.allow_resource_overcommit
                ):
                    samples = _resource_limited_samples(
                        active_modes,
                        reason="temporary_storage",
                        required_bytes=required_bytes,
                        available_bytes=available_bytes,
                    )
                else:
                    samples = [
                        _sample_in_subprocess(
                            args, total, inner, target_count, threshold, mode
                        )
                        for _ in range(args.repeats)
                        for mode in active_modes
                    ]
                summary = {
                    "total_threads": total,
                    "numba_threads": inner,
                    "joblib_workers": min(total // inner, target_count),
                    "target_count": target_count,
                    "min_logerr": threshold,
                    "input_suite": "heterogeneous" if args.coverage else "pwm",
                    "skewed_rows": args.skewed_rows,
                    "samples": samples,
                }
                for mode in active_modes:
                    mode_samples = [
                        sample
                        for sample in samples
                        if sample["mode"] == mode and sample.get("status") != "resource_limited"
                    ]
                    if not mode_samples:
                        limited = [
                            sample
                            for sample in samples
                            if sample["mode"] == mode
                            and sample.get("status") == "resource_limited"
                        ]
                        summary[f"{mode}_status"] = "resource_limited"
                        if limited:
                            for field in ("reason", "required_bytes", "available_bytes"):
                                if field in limited[0]:
                                    summary[f"{mode}_{field}"] = limited[0][field]
                        continue
                    values = [sample["wall_s"] for sample in mode_samples]
                    summary[f"{mode}_wall_s_median"] = statistics.median(values)
                    summary[f"{mode}_targets_per_s"] = target_count / summary[f"{mode}_wall_s_median"]
                    for field in (
                        "cache_bytes",
                        "parent_lifetime_max_rss_bytes",
                        "aggregate_peak_rss_bytes",
                    ):
                        field_values = [sample[field] for sample in mode_samples if field in sample]
                        if field_values:
                            summary[f"{mode}_{field}_median"] = statistics.median(field_values)
                results.append(summary)
                _write_benchmark_output(args.output, results)

    serial = {
        (result["target_count"], result["min_logerr"], mode): result[f"{mode}_wall_s_median"]
        for result in results
        if result["total_threads"] == 1 and result["numba_threads"] == 1
        for mode in modes
        if mode != "cold_jit_cli"
        if f"{mode}_wall_s_median" in result
    }
    for result in results:
        for mode in modes:
            if mode == "cold_jit_cli":
                continue
            if f"{mode}_wall_s_median" not in result:
                continue
            baseline = serial.get((result["target_count"], result["min_logerr"], mode))
            if baseline is not None:
                result[f"{mode}_speedup_vs_serial"] = baseline / result[f"{mode}_wall_s_median"]
    payload = json.dumps(_benchmark_payload(results), indent=2)
    _write_benchmark_output(args.output, results)
    print(payload)


if __name__ == "__main__":
    main()
