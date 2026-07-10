"""Measure fused profile alignment scaling across Numba thread budgets."""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import config as numba_config
from numba import set_num_threads

from mimosa.functions.alignment import build_anchor_csr, make_alignment_workspace, score_shift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()

    maximum = int(numba_config.NUMBA_NUM_THREADS)
    budgets = sorted({int(value) for value in args.threads.split(",") if 0 < int(value) <= maximum})
    rng = np.random.default_rng(args.seed)
    query = rng.random((args.rows, args.width), dtype=np.float32) * np.float32(5.0)
    target = rng.random((args.rows, args.width), dtype=np.float32) * np.float32(5.0)
    lengths = np.full(args.rows, args.width, dtype=np.int32)
    rows = np.repeat(np.arange(args.rows, dtype=np.int32), args.width)
    positions = np.tile(np.arange(args.width, dtype=np.int32), args.rows)
    anchors = build_anchor_csr(rows, positions, args.rows)

    print("rows\twidth\tthreads\twarm_ms\tspeedup\tscore\tn_sites")
    serial_ms = None
    for threads in budgets:
        set_num_threads(threads)
        workspace = make_alignment_workspace(args.rows, args.width)
        for generation in (1, 2):
            score_shift(
                query,
                lengths,
                target,
                lengths,
                anchors,
                anchors,
                0,
                5,
                3,
                "co",
                workspace,
                generation,
                threads > 1,
            )
        started = time.perf_counter()
        score, n_sites = score_shift(
            query,
            lengths,
            target,
            lengths,
            anchors,
            anchors,
            0,
            5,
            3,
            "co",
            workspace,
            3,
            threads > 1,
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if serial_ms is None:
            serial_ms = elapsed_ms
        print(
            f"{args.rows}\t{args.width}\t{threads}\t{elapsed_ms:.3f}\t{serial_ms / elapsed_ms:.3f}"
            f"\t{score:.8f}\t{n_sites}"
        )


if __name__ == "__main__":
    main()
