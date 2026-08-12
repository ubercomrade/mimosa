"""CLI: argparse adapter; stdout JSON, stderr diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .arrays import EncodedSequences
from .cache import Cache, clearcache
from .errors import MimosaError
from .io.bundles import write_null_bundle
from .io.fasta import MAX_FASTA_TOTAL_BASES, read_fasta, read_scores
from .io.readers import read_model
from .models import PWM, BaMM, Dimont, SiteGA, Slim
from .profiles.prepared import ScoreProfile

PROFILE_METRICS = ["co", "dice", "cosine"]
MODEL_TYPE_MAP = {
    "pwm": PWM,
    "bamm": BaMM,
    "sitega": SiteGA,
    "dimont": Dimont,
    "slim": Slim,
}
PROFILE_MODEL_TYPES = ["scores", *MODEL_TYPE_MAP]
MAX_GENERATED_SEQUENCES = 1_000_000
MAX_GENERATED_SEQUENCE_LENGTH = 1_000_000


def _bounded_positive_integer(name, maximum):
    def parse(value):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer.") from exc
        if parsed < 1 or parsed > maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between 1 and {maximum}."
            )
        return parsed

    return parse


def _validate_generated_dimensions(num_sequences, seq_length):
    if not isinstance(num_sequences, int) or isinstance(num_sequences, bool):
        raise ValueError("num_sequences must be an integer.")
    if not isinstance(seq_length, int) or isinstance(seq_length, bool):
        raise ValueError("seq_length must be an integer.")
    if not 1 <= num_sequences <= MAX_GENERATED_SEQUENCES:
        raise ValueError(
            f"num_sequences must be between 1 and {MAX_GENERATED_SEQUENCES}."
        )
    if not 1 <= seq_length <= MAX_GENERATED_SEQUENCE_LENGTH:
        raise ValueError(
            f"seq_length must be between 1 and {MAX_GENERATED_SEQUENCE_LENGTH}."
        )
    if num_sequences * seq_length > MAX_FASTA_TOTAL_BASES:
        raise ValueError(
            f"generated sequence bases must not exceed {MAX_FASTA_TOTAL_BASES}."
        )


def _read_typed_model(path, model_type, background=0.25):
    if model_type == "scores":
        return read_scores(path)
    if model_type not in MODEL_TYPE_MAP:
        raise MimosaError(f"unknown model type: {model_type}")
    kwargs = {"background": background}
    if model_type != "pwm":
        kwargs["format"] = model_type
    model = read_model(path, **kwargs)
    expected = MODEL_TYPE_MAP[model_type]
    if not isinstance(model, expected):
        raise MimosaError(
            f"model at '{path}' is {type(model).__name__}, not the requested {model_type} type."
        )
    return model


def _resolve_sequences(fasta_path, num_sequences, seq_length, seed):
    if fasta_path is not None:
        batch, _ = read_fasta(fasta_path)
        return batch
    _validate_generated_dimensions(num_sequences, seq_length)
    rng = np.random.default_rng(seed)
    return EncodedSequences(
        rng.integers(0, 4, size=num_sequences * seq_length, dtype=np.uint8),
        np.arange(num_sequences + 1, dtype=np.int64) * seq_length,
    )


def _validate_null_compatibility(dist, *, metric, sequences, background, search_range, window_radius, realign_window, min_logerr, model_types):
    from .io.bundles import sequence_fingerprint
    from .statistics import ALIGNMENT_VERSION

    expected_sequences = "none" if sequences is None else sequence_fingerprint(sequences)
    expected_background = "none" if background is None else sequence_fingerprint(background)
    checks = [
        (dist.strategy, "profile", "null distribution strategy is incompatible with profile comparison."),
        (dist.metric, metric, f"null distribution metric '{dist.metric}' is incompatible with requested metric '{metric}'."),
        (dist.contract["search_range"], search_range, "null distribution search range is incompatible with this comparison."),
        (dist.contract["window_radius"], window_radius, "null distribution window radius is incompatible with this comparison."),
        (dist.contract["realign_window"], realign_window, "null distribution realignment window is incompatible with this comparison."),
        (dist.contract["min_logerr"], np.float32(min_logerr), "null distribution minimum log-ERR is incompatible with this comparison."),
        (dist.sequence_fingerprint, expected_sequences, "null distribution sequence fingerprint is incompatible with this comparison."),
        (dist.background_fingerprint, expected_background, "null distribution background fingerprint is incompatible with this comparison."),
    ]
    for actual, expected, msg in checks:
        if actual != expected:
            raise MimosaError(msg)
    if not all(t == dist.model_type for t in model_types):
        raise MimosaError(
            f"null distribution model type '{dist.model_type}' is incompatible with compared model types '{', '.join(model_types)}'."
        )
    if dist.contract["alignment_version"] != ALIGNMENT_VERSION:
        raise MimosaError("null distribution alignment version is incompatible with this comparison.")


def _comparison_inputs(args):
    query_type = args.query_type
    target_type = args.target_type
    query = _read_typed_model(args.query, query_type, args.background_freq)
    target_paths = args.targets if isinstance(args.targets, list) else [args.targets]
    targets = [
        _read_typed_model(path, target_type, args.background_freq)
        for path in target_paths
    ]
    cache = Cache(args.cache_dir) if args.cache_dir else None

    sequences = None
    bg_sequences = None
    if not (
        isinstance(query, ScoreProfile)
        and all(isinstance(target, ScoreProfile) for target in targets)
    ):
        sequences = _resolve_sequences(args.fasta, args.num_sequences, args.seq_length, args.seed)
        bg_sequences = read_fasta(args.background)[0] if args.background else None
    return query, targets, sequences, bg_sequences, cache


def _annotate_comparison_results(results, args, sequences, background, model_types):
    if not args.pvalue:
        return results
    if not args.null_distribution:
        raise MimosaError("--pvalue requires an explicit --null-distribution bundle.")
    from .io.bundles import read_null_bundle
    from .statistics import NullDistribution, annotate_results

    dist = NullDistribution(**read_null_bundle(args.null_distribution))
    from .profiles.normalization import HybridEmpiricalLogTail, normalization_fingerprint

    if dist.contract["normalization_version"] != normalization_fingerprint(
        HybridEmpiricalLogTail()
    ):
        raise MimosaError("null distribution normalization is incompatible with this comparison.")
    _validate_null_compatibility(
        dist,
        metric=args.metric,
        sequences=sequences,
        background=background,
        search_range=args.search_range,
        window_radius=args.window_radius,
        realign_window=args.realign_window,
        min_logerr=args.min_logerr,
        model_types=model_types,
    )
    return annotate_results(
        results, dist, effective_number_of_targets=args.effective_number_of_targets
    )


def _run_compare(args):
    from .compare import compare
    from numba import set_num_threads

    set_num_threads(args.numba_threads)
    query, targets, sequences, background, cache = _comparison_inputs(args)
    result = compare(
        query,
        targets[0],
        sequences,
        metric=args.metric,
        search_range=args.search_range,
        window_radius=args.window_radius,
        realign_window=args.realign_window,
        min_logerr=args.min_logerr,
        background=background,
        cache=cache,
    )
    result = _annotate_comparison_results(
        [result],
        args,
        sequences,
        background,
        (args.query_type, args.target_type),
    )[0]
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_compare_many(args):
    from .compare import compare_many
    from numba import set_num_threads

    set_num_threads(args.numba_threads)
    query, targets, sequences, background, cache = _comparison_inputs(args)
    results = compare_many(
        query,
        targets,
        sequences,
        metric=args.metric,
        search_range=args.search_range,
        window_radius=args.window_radius,
        realign_window=args.realign_window,
        min_logerr=args.min_logerr,
        background=background,
        cache=cache,
        total_threads=args.total_threads,
        inner_threads=args.numba_threads,
    )
    results = _annotate_comparison_results(
        results,
        args,
        sequences,
        background,
        (args.query_type, args.target_type),
    )
    print(json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True))
    return 0


def _run_build_null(args):
    from .statistics import build_null

    if not os.path.isdir(args.motifs):
        raise MimosaError("motif collection path must be a directory.")
    files = sorted(
        filename for filename in os.listdir(args.motifs) if filename.lower().endswith(".meme")
    )
    if not files:
        raise MimosaError(f"no pwm files found in {args.motifs}.")
    models = [_read_typed_model(os.path.join(args.motifs, filename), "pwm") for filename in files]
    cache = Cache(args.cache_dir) if args.cache_dir else None
    sequences = _resolve_sequences(args.fasta, args.num_sequences, args.seq_length, args.seed)
    background = read_fasta(args.background)[0] if args.background else None
    dist = build_null(
        models,
        sequences=sequences,
        background=background,
        metric=args.metric,
        n_samples=args.num_samples,
        seed=args.seed,
        search_range=args.search_range,
        window_radius=args.window_radius,
        realign_window=args.realign_window,
        min_logerr=args.min_logerr,
        cache=cache,
    )
    write_null_bundle(args.output, dist)
    summary = {
        "output": args.output,
        "n_models": len(models),
        "n_null": dist.n_null,
        "model_type": dist.model_type,
        "shuffle": True,
        "seed": dist.seed,
        "metric": dist.metric,
        "strategy": dist.strategy,
        "estimator": "empirical_upper_tail",
        "normalization": dist.contract["normalization_version"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_cache(args):
    cache_dir = args.cache_dir
    root = os.path.abspath(cache_dir)
    if root == os.path.dirname(root) or root == os.path.expanduser("~"):
        raise MimosaError("--cache-dir points to a dangerously broad directory.")
    if os.path.exists(root) and (not os.path.isdir(root) or os.path.islink(root)):
        raise MimosaError("--cache-dir must be a real directory, not a file or symlink.")
    removed = clearcache(Cache(cache_dir))
    print(json.dumps({"cache_dir": cache_dir, "removed": removed}, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mimosa", description="Motif comparison and statistical evaluation."
    )
    parser.add_argument("--version", "-V", action="version", version=f"mimosa {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_compare_arguments(command, many):
        p = sub.add_parser(command, help="compare motif score profiles")
        p.add_argument("query", help="query model or score-profile file")
        p.add_argument(
            "targets",
            nargs="+" if many else None,
            help="target model or score-profile file(s)",
        )
        p.add_argument("--query-type", required=True, choices=PROFILE_MODEL_TYPES)
        p.add_argument("--target-type", required=True, choices=PROFILE_MODEL_TYPES)
        if many:
            p.add_argument("--total-threads", type=int, default=1)
        p.add_argument(
            "--numba-threads", type=int, choices=range(1, 5), default=1
        )
        p.add_argument("--metric", default="co", choices=PROFILE_METRICS)
        p.add_argument("--search-range", type=int, default=10)
        p.add_argument("--window-radius", type=int, default=10)
        p.add_argument("--realign-window", type=int, default=3)
        p.add_argument("--min-logerr", type=float, default=0.0)
        p.add_argument("--fasta")
        p.add_argument("--background")
        p.add_argument(
            "--num-sequences",
            type=_bounded_positive_integer("num-sequences", MAX_GENERATED_SEQUENCES),
            default=1000,
        )
        p.add_argument(
            "--seq-length",
            type=_bounded_positive_integer("seq-length", MAX_GENERATED_SEQUENCE_LENGTH),
            default=200,
        )
        p.add_argument("--seed", type=int, default=127)
        p.add_argument("--background-freq", type=float, default=0.25)
        p.add_argument("--cache-dir")
        p.add_argument("--null-distribution")
        p.add_argument(
            "--effective-number-of-targets",
            type=_bounded_positive_integer("effective-number-of-targets", MAX_GENERATED_SEQUENCES),
        )
        p.add_argument("--pvalue", action="store_true")

    add_compare_arguments("compare", False)
    add_compare_arguments("compare-many", True)

    p = sub.add_parser("build-null", help="build a null distribution from motif comparisons")
    p.add_argument("motifs", help="motif collection path")
    p.add_argument("--output", required=True)
    p.add_argument("--metric", default="co", choices=PROFILE_METRICS)
    p.add_argument("--fasta")
    p.add_argument("--background")
    p.add_argument(
        "--num-sequences",
        type=_bounded_positive_integer("num-sequences", MAX_GENERATED_SEQUENCES),
        default=1000,
    )
    p.add_argument(
        "--seq-length",
        type=_bounded_positive_integer("seq-length", MAX_GENERATED_SEQUENCE_LENGTH),
        default=200,
    )
    p.add_argument("--seed", type=int, default=127)
    p.add_argument(
        "--num-samples",
        type=_bounded_positive_integer("num-samples", 1_000_000),
        default=2000,
    )
    p.add_argument("--search-range", type=int, default=10)
    p.add_argument("--window-radius", type=int, default=10)
    p.add_argument("--realign-window", type=int, default=3)
    p.add_argument("--min-logerr", type=float, default=0.0)
    p.add_argument("--cache-dir")

    p = sub.add_parser("cache", help="manage the disk cache")
    p.add_argument("operation", choices=("clear",), help="cache operation (clear)")
    p.add_argument("--cache-dir", default=".mimosa-cache")
    return parser


def _validate_cli_arguments(args):
    if args.command not in {"compare", "compare-many"}:
        return
    scores_only = args.query_type == args.target_type == "scores"
    if scores_only and (args.fasta is not None or args.background is not None):
        raise MimosaError("--fasta and --background are not used for scores-only comparison.")
    if not args.pvalue and args.null_distribution is not None:
        raise MimosaError("--null-distribution requires --pvalue.")
    if not args.pvalue and args.effective_number_of_targets is not None:
        raise MimosaError("--effective-number-of-targets requires --pvalue.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        _validate_cli_arguments(args)
        if args.command == "compare":
            return _run_compare(args)
        if args.command == "compare-many":
            if args.total_threads < 1:
                raise ValueError("total_threads must be a positive integer.")
            if args.total_threads % args.numba_threads:
                raise ValueError("total_threads must be divisible by numba-threads.")
            return _run_compare_many(args)
        if args.command == "build-null":
            return _run_build_null(args)
        if args.command == "cache":
            return _run_cache(args)
        raise MimosaError(f"unknown command: {args.command}")
    except (MimosaError, ValueError, TypeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
