"""Benchmark model-group scan dispatch on the real foreground fixture."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _repeat_batch(batch, count):
    from mimosa import EncodedSequences

    rows = [batch[index] for index in range(len(batch))]
    return EncodedSequences.from_rows(
        [rows[index % len(rows)] for index in range(count)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-count", type=int, default=10_000)
    parser.add_argument("--groups", default="1,2,4,16,64")
    parser.add_argument("--threads", default="1,2,4,6,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = [int(value) for value in args.groups.split(",") if value]
    threads = [int(value) for value in args.threads.split(",") if value]
    if args.sequence_count < 1 or args.repeats < 1:
        raise SystemExit("sequence-count and repeats must be positive")
    if not groups or not threads or any(value < 1 for value in (*groups, *threads)):
        raise SystemExit("groups and threads must contain positive integers")
    os.environ["NUMBA_NUM_THREADS"] = str(max(threads))

    from numba import set_num_threads
    from mimosa import PWM
    from mimosa.io.fasta import read_fasta
    from mimosa.io.models import read_meme
    from mimosa.models import pwm_from_pfm
    from mimosa.parallel import scan_dispatch_path
    from mimosa.scan import _scan_models_batch, _scan_offsets

    name, pfm = read_meme(str(ROOT / "examples" / "foxa2.meme"))
    base = pwm_from_pfm(pfm, name=name)
    source, _ = read_fasta(str(ROOT / "examples" / "foreground.fa"))
    sequences = _repeat_batch(source, args.sequence_count)
    results = []
    for thread_count in threads:
        set_num_threads(thread_count)
        for group_size in groups:
            models = [
                PWM(f"{base.name}#{index}", base.weights, base.background)
                for index in range(group_size)
            ]
            items = int(_scan_offsets(sequences, models[0])[-1])
            path = scan_dispatch_path(items, rows=len(sequences), groups=group_size)
            samples = []
            _scan_models_batch(models, sequences)
            for _ in range(args.repeats):
                started = time.perf_counter()
                _scan_models_batch(models, sequences)
                samples.append(time.perf_counter() - started)
            results.append(
                {
                    "threads": thread_count,
                    "group_size": group_size,
                    "dispatch_path": path,
                    "median_s": statistics.median(samples),
                    "samples_s": samples,
                }
            )
    payload = json.dumps({"benchmark": "mimosa-scan-dispatch-v1", "results": results}, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
