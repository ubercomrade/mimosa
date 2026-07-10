"""Reproducible TSV benchmark for fused profile alignment."""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np
from numba import set_num_threads

from mimosa.functions.alignment import build_anchor_csr, make_alignment_workspace, score_shift

CASES = {
    "small": (100, 100),
    "medium": (1_000, 100),
    "large": (4_000, 100),
    "collection": (1_000, 100),
}
METRICS = ("co", "dice", "co_rowwise", "dice_rowwise", "cosine")


def _anchors(scores: np.ndarray, mode: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if mode == "best":
        rows = np.arange(scores.shape[0], dtype=np.int32)
        positions = np.argmax(scores, axis=1).astype(np.int32)
    else:
        rows, positions = np.nonzero(scores >= threshold)
        rows = rows.astype(np.int32)
        positions = positions.astype(np.int32)
    return build_anchor_csr(rows, positions, scores.shape[0])


def _run_case(  # noqa: PLR0913
    rows: int,
    width: int,
    metric: str,
    anchor_mode: str,
    threshold: float,
    orientations: int,
    threads: int,
    seed: int,
) -> tuple[float, float, float, float, int, int, str]:
    rng = np.random.default_rng(seed)
    query = rng.random((rows, width), dtype=np.float32) * np.float32(5.0)
    target = rng.random((rows, width), dtype=np.float32) * np.float32(5.0)
    lengths = np.full(rows, width, dtype=np.int32)
    query_anchors = _anchors(query, anchor_mode, threshold)
    target_anchors = _anchors(target, anchor_mode, threshold)
    use_parallel = threads > 1
    set_num_threads(threads)

    def measure() -> tuple[float, float, int, int, str]:
        started = time.perf_counter()
        best_score = 0.0
        best_sites = 0
        best_shift = 0
        best_orientation = "++"
        for orientation in range(orientations):
            workspace = make_alignment_workspace(rows, width)
            oriented_target = target if orientation % 2 == 0 else target[:, ::-1].copy()
            for generation, shift in enumerate(range(-5, 6), start=1):
                score, n_sites = score_shift(
                    query,
                    lengths,
                    oriented_target,
                    lengths,
                    query_anchors,
                    target_anchors,
                    shift,
                    5,
                    3,
                    metric,
                    workspace,
                    generation,
                    use_parallel,
                )
                if score > best_score:
                    best_score = score
                    best_sites = n_sites
                    best_shift = shift
                    best_orientation = ("++", "--", "+-", "-+")[orientation]
        return (time.perf_counter() - started) * 1_000.0, best_score, best_sites, best_shift, best_orientation

    cold_ms, _, _, _, _ = measure()
    warm_ms, score, n_sites, best_shift, best_orientation = measure()
    peak_rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return cold_ms, warm_ms, peak_rss_kb, score, n_sites, best_shift, best_orientation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="small")
    parser.add_argument("--metric", choices=(*METRICS, "all"), default="all")
    parser.add_argument("--anchor-mode", choices=("best", "threshold", "all"), default="all")
    parser.add_argument("--threshold", type=float, default=4.5)
    parser.add_argument("--orientations", type=int, choices=(1, 4), default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()

    rows, width = CASES[args.case]
    metrics = METRICS if args.metric == "all" else (args.metric,)
    modes = ("best", "threshold") if args.anchor_mode == "all" else (args.anchor_mode,)
    print(
        "case\trows\twidth\tmetric\tanchors\tthreads\torientations\tcold_ms\twarm_ms\tpeak_rss_kb"
        "\tscore\tshift\torientation\tn_sites"
    )
    for metric in metrics:
        for mode in modes:
            cold_ms, warm_ms, peak_rss_kb, score, n_sites, best_shift, orientation = _run_case(
                rows,
                width,
                metric,
                mode,
                args.threshold,
                args.orientations,
                args.threads,
                args.seed,
            )
            print(
                f"{args.case}\t{rows}\t{width}\t{metric}\t{mode}\t{args.threads}\t{args.orientations}"
                f"\t{cold_ms:.3f}\t{warm_ms:.3f}\t{peak_rss_kb:.0f}\t{score:.8f}\t{best_shift}"
                f"\t{orientation}\t{n_sites}"
            )


if __name__ == "__main__":
    main()
