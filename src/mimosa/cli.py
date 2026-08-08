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
from .compare import compare
from .errors import MimosaError
from .io.bundles import write_null_bundle
from .io.fasta import read_fasta, read_scores
from .io.models import read_meme
from .io.readers import read_model
from .models import PWM, pwm_from_pfm
from .models import BaMM, Dimont, SiteGA, Slim
from .profiles.normalization import HybridEmpiricalLogTail
from .profiles.prepared import ScoreProfile
from .statistics import annotate_results, build_null

MODEL_TYPES = ["pwm", "bamm", "sitega", "dimont", "slim"]
PROFILE_MODEL_TYPES = ["scores", *MODEL_TYPES]
PROFILE_METRICS = ["co", "dice", "cosine"]
MODEL_TYPE_MAP = {
    "pwm": PWM,
    "bamm": BaMM,
    "sitega": SiteGA,
    "dimont": Dimont,
    "slim": Slim,
}


class CLIError(Exception):
    pass


def _read_typed_model(path, model_type, background=0.25):
    if model_type == "scores":
        return read_scores(path)
    if model_type not in MODEL_TYPE_MAP:
        raise CLIError(f"unknown model type: {model_type}")
    kwargs = {"background": background}
    if model_type != "pwm":
        kwargs["format"] = model_type
    model = read_model(path, **kwargs)
    expected = MODEL_TYPE_MAP[model_type]
    if not isinstance(model, expected):
        raise CLIError(
            f"model at '{path}' is {type(model).__name__}, not the requested {model_type} type."
        )
    return model


def _resolve_sequences(fasta_path, num_sequences, seq_length, seed):
    if fasta_path is not None:
        batch, _ = read_fasta(fasta_path)
        return batch
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_sequences):
        rows.append(rng.integers(0, 4, size=seq_length, dtype=np.uint8))
    return EncodedSequences.from_rows(rows)


def _read_model_collection(path, model_type):
    if os.path.isdir(path):
        if model_type == "pwm":
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(".meme"))
        elif model_type == "bamm":
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(".ihbcp"))
        elif model_type == "sitega":
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(".mat"))
        elif model_type in ("dimont", "slim"):
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(".xml"))
        else:
            raise CLIError(f"unsupported model type for directory: {model_type}")
        if not files:
            raise CLIError(f"no {model_type} files found in {path}.")
        return [_read_typed_model(os.path.join(path, f), model_type) for f in files]
    if model_type == "pwm":
        models = []
        idx = 0
        while True:
            try:
                name, pfm = read_meme(path, index=idx)
                models.append(pwm_from_pfm(pfm, background=0.25, name=name))
                idx += 1
            except MimosaError as e:
                if "out of range" in str(e):
                    break
                raise
        if not models:
            raise CLIError(f"no motifs found in {path}.")
        return models
    return [_read_typed_model(path, model_type)]


def _validate_null_compatibility(dist, *, strategy, metric, sequences, background, search_range, window_radius, realign_window, min_logerr, model_types=None):
    d = dist.to_dict() if hasattr(dist, "to_dict") and not isinstance(dist, dict) else dist
    from .io.bundles import sequence_fingerprint

    expected_sequences = "none" if sequences is None else sequence_fingerprint(sequences)
    expected_background = "none" if background is None else sequence_fingerprint(background)
    checks = [
        (d["strategy"], strategy, f"null distribution strategy '{d['strategy']}' is incompatible with {strategy} comparison."),
        (d["metric"], metric, f"null distribution metric '{d['metric']}' is incompatible with requested metric '{metric}'."),
        (d["contract"]["search_range"], search_range, "null distribution search range is incompatible with this comparison."),
        (d["contract"]["window_radius"], window_radius, "null distribution window radius is incompatible with this comparison."),
        (d["contract"]["realign_window"], realign_window, "null distribution realignment window is incompatible with this comparison."),
        (d["contract"]["min_logerr"], np.float32(min_logerr), "null distribution minimum log-ERR is incompatible with this comparison."),
        (d["sequence_fingerprint"], expected_sequences, "null distribution sequence fingerprint is incompatible with this comparison."),
        (d["background_fingerprint"], expected_background, "null distribution background fingerprint is incompatible with this comparison."),
    ]
    for actual, expected, msg in checks:
        if actual != expected:
            raise CLIError(msg)
    if model_types is not None and not all(t == d["model_type"] for t in model_types):
        raise CLIError(
            f"null distribution model type '{d['model_type']}' is incompatible with compared model types '{', '.join(model_types)}'."
        )


def _annotate_cli_result(result, args, *, strategy, metric, sequences, background, search_range, window_radius, realign_window, min_logerr, model_types=None):
    if not args.pvalue:
        return result
    if not args.null_distribution:
        raise CLIError("--pvalue requires an explicit --null-distribution bundle.")
    from .io.bundles import read_null_bundle
    from .statistics import NullDistribution

    dist = NullDistribution(**read_null_bundle(args.null_distribution))
    if dist.contract["normalization_version"] != "hybrid-log-tail-v2;bins=65536":
        raise CLIError("null distribution normalization is incompatible with this comparison.")
    _validate_null_compatibility(
        dist,
        strategy=strategy,
        metric=metric,
        sequences=sequences,
        background=background,
        search_range=search_range,
        window_radius=window_radius,
        realign_window=realign_window,
        min_logerr=min_logerr,
        model_types=model_types,
    )
    effective = args.effective_number_of_targets
    return annotate_results([result], dist, effective_number_of_targets=effective)[0]


def _run_profile(args):
    type1, type2 = args.model1_type, args.model2_type
    if type1 not in PROFILE_MODEL_TYPES:
        raise CLIError(f"--model1-type must be one of: {', '.join(PROFILE_MODEL_TYPES)}")
    if type2 not in PROFILE_MODEL_TYPES:
        raise CLIError(f"--model2-type must be one of: {', '.join(PROFILE_MODEL_TYPES)}")
    if args.metric not in PROFILE_METRICS:
        raise CLIError(f"--metric must be one of: {', '.join(PROFILE_METRICS)}")

    model1 = _read_typed_model(args.model1, type1, args.background_freq)
    model2 = _read_typed_model(args.model2, type2, args.background_freq)
    cache = Cache(args.cache_dir) if args.cache_dir else None
    normalization = HybridEmpiricalLogTail()

    sequences = None
    bg_sequences = None
    if isinstance(model1, ScoreProfile) and isinstance(model2, ScoreProfile):
        result = compare(
            model1,
            model2,
            metric=args.metric,
            search_range=args.search_range,
            window_radius=args.window_radius,
            realign_window=args.realign_window,
            min_logerr=args.min_logerr,
            normalization=normalization,
            cache=cache,
        )
    else:
        sequences = _resolve_sequences(args.fasta, args.num_sequences, args.seq_length, args.seed)
        bg_sequences = read_fasta(args.background)[0] if args.background else None
        result = compare(
            model1,
            model2,
            sequences,
            metric=args.metric,
            search_range=args.search_range,
            window_radius=args.window_radius,
            realign_window=args.realign_window,
            min_logerr=args.min_logerr,
            background=bg_sequences,
            normalization=normalization,
            cache=cache,
        )

    annotated = _annotate_cli_result(
        result,
        args,
        strategy="profile",
        metric=args.metric,
        sequences=sequences,
        background=bg_sequences,
        search_range=args.search_range,
        window_radius=args.window_radius,
        realign_window=args.realign_window,
        min_logerr=args.min_logerr,
        model_types=(type1, type2),
    )
    print(json.dumps(annotated.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_build_null(args):
    if args.model_type != "pwm":
        raise CLIError("build-null requires --model-type pwm.")
    if not os.path.isdir(args.motifs):
        raise CLIError("motif collection path must be a directory.")
    if args.metric not in PROFILE_METRICS:
        raise CLIError(f"--metric must be one of: {', '.join(PROFILE_METRICS)}")
    models = _read_model_collection(args.motifs, args.model_type)
    cache = Cache(args.cache_dir) if args.cache_dir else None
    sequences = _resolve_sequences(args.fasta, args.num_sequences, args.seq_length, args.seed)
    dist = build_null(
        models,
        sequences=sequences,
        metric=args.metric,
        n_samples=args.num_samples,
        seed=args.seed,
        search_range=args.search_range,
        window_radius=args.window_radius,
        realign_window=args.realign_window,
        min_logerr=args.min_logerr,
        normalization=HybridEmpiricalLogTail(),
        cache=cache,
    )
    write_null_bundle(args.output, dist)
    summary = {
        "output": args.output,
        "n_models": len(models),
        "n_comparisons": dist.n_null,
        "n_null": dist.n_null,
        "model_type": dist.model_type,
        "shuffle": True,
        "seed": dist.seed,
        "metric": dist.metric,
        "strategy": dist.strategy,
        "estimator": "empirical_upper_tail",
        "normalization": dist.contract["normalization_version"],
    }
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_cache(args):
    if args.operation != "clear":
        raise CLIError("cache requires a subcommand: clear")
    cache_dir = args.cache_dir
    root = os.path.abspath(cache_dir)
    if root == os.path.dirname(root) or root == os.path.expanduser("~"):
        raise CLIError("--cache-dir points to a dangerously broad directory.")
    if os.path.exists(root) and (not os.path.isdir(root) or os.path.islink(root)):
        raise CLIError("--cache-dir must be a real directory, not a file or symlink.")
    removed = clearcache(Cache(cache_dir))
    if not args.quiet:
        print(json.dumps({"cache_dir": cache_dir, "removed": removed}, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mimosa", description="Motif comparison and statistical evaluation."
    )
    parser.add_argument("--version", "-V", action="version", version=f"mimosa {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="compare two motif score profiles")
    p.add_argument("model1", help="first model or score-profile file")
    p.add_argument("model2", help="second model or score-profile file")
    p.add_argument("--model1-type", required=True)
    p.add_argument("--model2-type", required=True)
    p.add_argument("--metric", default="co")
    p.add_argument("--search-range", type=int, default=10)
    p.add_argument("--window-radius", type=int, default=10)
    p.add_argument("--realign-window", type=int, default=3)
    p.add_argument("--min-logerr", type=float, default=0.0)
    p.add_argument("--fasta")
    p.add_argument("--background")
    p.add_argument("--num-sequences", type=int, default=1000)
    p.add_argument("--seq-length", type=int, default=200)
    p.add_argument("--seed", type=int, default=127)
    p.add_argument("--background-freq", type=float, default=0.25)
    p.add_argument("--cache-dir")
    p.add_argument("--null-distribution")
    p.add_argument("--effective-number-of-targets", type=int)
    p.add_argument("--pvalue", action="store_true")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("build-null", help="build a null distribution from motif comparisons")
    p.add_argument("motifs", help="motif collection path")
    p.add_argument("--model-type", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric", default="co")
    p.add_argument("--fasta")
    p.add_argument("--num-sequences", type=int, default=1000)
    p.add_argument("--seq-length", type=int, default=200)
    p.add_argument("--seed", type=int, default=127)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--search-range", type=int, default=10)
    p.add_argument("--window-radius", type=int, default=10)
    p.add_argument("--realign-window", type=int, default=3)
    p.add_argument("--min-logerr", type=float, default=0.0)
    p.add_argument("--cache-dir")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("cache", help="manage the disk cache")
    p.add_argument("operation", help="cache operation (clear)")
    p.add_argument("--cache-dir", default=".mimosa-cache")
    p.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "profile":
            return _run_profile(args)
        if args.command == "build-null":
            return _run_build_null(args)
        if args.command == "cache":
            return _run_cache(args)
        raise CLIError(f"unknown command: {args.command}")
    except CLIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (MimosaError, ValueError, TypeError, OSError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
