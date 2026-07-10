"""Compare old and fused profile orientation scorers on identical inputs."""

from __future__ import annotations

import argparse
import inspect
import resource
import time
import tracemalloc

import numpy as np

from mimosa.comparison import profile
from mimosa.comparison.config import create_comparator_config

FUSED_PARAMETER_COUNT = 7


def _anchor_csr(rows: np.ndarray, positions: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(rows, minlength=n_rows)
    offsets = np.empty(n_rows + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return np.ascontiguousarray(positions, dtype=np.int32), offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    values1 = rng.random((2, args.rows, args.width), dtype=np.float32) * np.float32(5.0)
    values2 = rng.random((2, args.rows, args.width), dtype=np.float32) * np.float32(5.0)
    lengths = np.full(args.rows, args.width, dtype=np.int64)
    bundle1 = {"values": values1, "lengths": lengths, "padding_value": 0.0}
    bundle2 = {"values": values2, "lengths": lengths, "padding_value": 0.0}
    cfg = create_comparator_config(
        metric="co",
        search_range=5,
        window_radius=5,
        realign_window=3,
        min_logfpr=args.threshold,
        n_jobs=1,
    )
    raw_query = profile._collect_anchor_sites(values1[0], lengths, args.threshold)
    raw_target = profile._collect_anchor_sites(values2[0], lengths, args.threshold)
    parameter_count = len(inspect.signature(profile._score_profile_orientation_pair).parameters)

    def score():
        if parameter_count == FUSED_PARAMETER_COUNT:
            return profile._score_profile_orientation_pair(
                bundle1,
                bundle2,
                0,
                0,
                _anchor_csr(*raw_query, args.rows),
                _anchor_csr(*raw_target, args.rows),
                cfg,
            )
        radius = int(cfg["window_radius"])
        return profile._score_profile_orientation_pair(
            bundle1,
            bundle2,
            0,
            0,
            profile._build_window_offsets(radius),
            -radius,
            radius,
            raw_query,
            raw_target,
            cfg,
        )

    score()
    tracemalloc.start()
    started = time.perf_counter()
    result = score()
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    _, allocated_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("label\trows\twidth\twarm_ms\tpeak_rss_kb\tallocation_peak_kb\tscore\tshift\tn_sites")
    print(
        f"{args.label}\t{args.rows}\t{args.width}\t{elapsed_ms:.3f}\t{peak_rss_kb}\t{allocated_peak / 1024.0:.1f}"
        f"\t{result['score']:.8f}\t{result['shift']}\t{result['n_sites']}"
    )


if __name__ == "__main__":
    main()
