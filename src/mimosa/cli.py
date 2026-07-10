import argparse
import json
import logging
import sys
from typing import Any, Dict

from mimosa.api import create_one_to_one_config, run_one_to_one
from mimosa.batches import make_random_sequence_batch
from mimosa.cache import clear_cache
from mimosa.comparison import (
    SUPPORTED_MOTIF_METRICS,
    SUPPORTED_PROFILE_METRICS,
    create_comparator_config,
    validate_metric,
)
from mimosa.io import read_fasta
from mimosa.models import read_models
from mimosa.nulls import (
    NullBuildRequest,
    file_fingerprint,
    parse_group_relations,
    run_build_null_request,
)
from mimosa.progress import TqdmLoggingHandler, should_enable_progress
from mimosa.validation import validate_file_exists, validate_positive_int

PROFILE_MODEL_TYPES = ["scores", "pwm", "bamm", "sitega", "dimont", "slim"]
MOTIF_MODEL_TYPES = ["pwm", "bamm", "sitega", "dimont", "slim"]


def setup_logging(verbose: bool, progress: bool | None = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handler: logging.Handler = TqdmLoggingHandler() if should_enable_progress(progress) else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="MIMOSA: Compare motifs in `profile` and `motif` modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare precomputed score profiles directly
  mimosa profile scores_1.fasta scores_2.fasta \
    --model1-type scores --model2-type scores --metric cosine

  # Compare motifs through sequence-derived profiles
  mimosa profile model1.meme model2.ihbcp \
    --model1-type pwm --model2-type bamm \
    --fasta sequences.fa --metric co --min-logfpr 2 --window-radius 10

  # Direct motif comparison (former tomtom-like mode)
  mimosa motif model1.meme model2.pfm \
    --model1-type pwm --model2-type pwm \
    --metric pcc

  # Build a pooled null distribution from unrelated motif comparisons
  mimosa build-null motifs.meme --model-type pwm --groups groups.tsv \
    --strategy motif --metric pcc --output motifs-pcc.null.joblib

        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    _add_profile_parser(subparsers)
    _add_motif_parser(subparsers)
    _add_build_null_parser(subparsers)
    _add_cache_parser(subparsers)

    return parser


def _add_input_file_arguments(
    parser: argparse.ArgumentParser,
    model_types: list[str],
    first_help: str,
    second_help: str,
) -> argparse._ArgumentGroup:
    """Add required model inputs and types to one parser."""
    parser.add_argument("model1", help=first_help)
    parser.add_argument("model2", help=second_help)
    io_group = parser.add_argument_group("Input Options")
    io_group.add_argument(
        "--model1-type",
        choices=model_types,
        required=True,
        help=f"Format of the first input. Choices: {', '.join(model_types)}.",
    )
    io_group.add_argument(
        "--model2-type",
        choices=model_types,
        required=True,
        help=f"Format of the second input. Choices: {', '.join(model_types)}.",
    )
    return io_group


def _add_sequence_generation_arguments(
    io_group: argparse._ArgumentGroup,
    *,
    fasta_help: str,
    num_sequences_default: int,
    seq_length_default: int,
    background_help: str | None = None,
) -> None:
    """Add FASTA- and random-sequence-related arguments."""
    io_group.add_argument("--fasta", help=fasta_help)
    if background_help is not None:
        io_group.add_argument("--background", help=background_help)
        io_group.add_argument("--promoters", dest="background", help=argparse.SUPPRESS)
    io_group.add_argument(
        "--num-sequences",
        type=int,
        default=num_sequences_default,
        help="Number of random sequences to generate when FASTA input is omitted. (default: %(default)s)",
    )
    io_group.add_argument(
        "--seq-length",
        type=int,
        default=seq_length_default,
        help="Length of random sequences generated when FASTA input is omitted. (default: %(default)s)",
    )


def _add_common_technical_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_seed: bool = True,
    include_jobs: bool = True,
    include_cache: bool = False,
) -> argparse._ArgumentGroup:
    """Add shared technical arguments for profile/motif parsers."""
    technical_group = parser.add_argument_group("Technical Options")
    technical_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    if include_seed:
        technical_group.add_argument(
            "--seed",
            type=int,
            default=127,
            help="Global random seed for reproducible stochastic steps. (default: %(default)s)",
        )
    if include_jobs:
        technical_group.add_argument(
            "--jobs",
            type=int,
            default=-1,
            help="Number of Numba threads per numerical kernel. Use -1 for the runtime maximum. (default: %(default)s)",
        )
    technical_group.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show progress bars on stderr. Defaults to auto-detecting interactive terminals.",
    )
    if include_cache:
        technical_group.add_argument(
            "--cache",
            choices=["off", "on"],
            default="off",
            help="Enable lazy disk cache for derived profiles. (default: %(default)s)",
        )
        technical_group.add_argument(
            "--cache-dir",
            default=".mimosa-cache",
            help="Directory used for lazy profile cache files. (default: %(default)s)",
        )
    return technical_group


def _add_significance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add stored null distribution file comparison options."""
    group = parser.add_argument_group("Significance Options")
    group.add_argument(
        "--pvalue",
        action="store_true",
        help="Annotate the result using a compatible stored null distribution.",
    )
    group.add_argument(
        "--null-distribution",
        help=(
            "Path to a trusted null distribution file built with 'mimosa build-null'. "
            "Joblib files must not be loaded from untrusted sources."
        ),
    )
    group.add_argument(
        "--null-search-dir",
        action="append",
        dest="null_search_dirs",
        help="Additional directory searched for compatible null distribution files.",
    )
    group.add_argument(
        "--effective-number-of-targets",
        type=int,
        help="Override the target count used for E-value calculation.",
    )


def _add_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the profile mode parser."""
    parser = subparsers.add_parser(
        "profile",
        help="Compare motifs via score profiles: either precomputed scores or profiles generated from motif scans.",
    )
    io_group = _add_input_file_arguments(
        parser,
        PROFILE_MODEL_TYPES,
        "Path to the first input model or score-profile file.",
        "Path to the second input model or score-profile file.",
    )
    _add_sequence_generation_arguments(
        io_group,
        fasta_help=(
            "Path to FASTA sequences used to scan motif inputs. "
            "If omitted and motif scanning is required, random sequences are generated."
        ),
        background_help=(
            "Optional FASTA sequences used to calibrate profile normalization. "
            "If omitted, normalization is fitted on the comparison sequences."
        ),
        num_sequences_default=1000,
        seq_length_default=200,
    )
    profile_group = parser.add_argument_group("Profile Comparison Options")
    profile_group.add_argument(
        "--metric",
        choices=list(SUPPORTED_PROFILE_METRICS),
        default="co",
        help=(
            "Window-based profile similarity metric. "
            f"Choices: {', '.join(SUPPORTED_PROFILE_METRICS)}. (default: %(default)s)"
        ),
    )
    profile_group.add_argument(
        "--search-range",
        type=int,
        default=10,
        help="Maximum site-center shift explored between motifs. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--window-radius",
        type=int,
        default=10,
        help="Radius of the site-centered comparison window in profile positions. (default: %(default)s)",
    )
    profile_group.add_argument(
        "--realign-window",
        type=int,
        default=3,
        help=(
            "Half-width of the local realignment window used for anchors from the second motif. (default: %(default)s)"
        ),
    )
    profile_group.add_argument(
        "--min-logfpr",
        type=float,
        default=None,
        help=(
            "Select all sites at or above this logFPR threshold. "
            "If omitted or set to 0, one best site per sequence is used."
        ),
    )

    _add_common_technical_arguments(parser, include_cache=True)
    _add_significance_arguments(parser)


def _add_motif_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the motif mode parser."""
    parser = subparsers.add_parser(
        "motif",
        help="Compare motifs directly by aligning their matrix or tensor representations.",
    )
    io_group = _add_input_file_arguments(
        parser,
        MOTIF_MODEL_TYPES,
        "Path to the first motif model file.",
        "Path to the second motif model file.",
    )
    _add_sequence_generation_arguments(
        io_group,
        fasta_help=(
            "Optional FASTA sequences used for PFM reconstruction. "
            "If omitted when reconstruction is required, random sequences are generated."
        ),
        num_sequences_default=20000,
        seq_length_default=100,
    )

    motif_group = parser.add_argument_group("Motif Comparison Options")
    motif_group.add_argument(
        "--metric",
        choices=list(SUPPORTED_MOTIF_METRICS),
        default="pcc",
        help=(f"Column-wise comparison metric. Choices: {', '.join(SUPPORTED_MOTIF_METRICS)}. (default: %(default)s)"),
    )
    motif_group.add_argument(
        "--pfm-mode",
        action="store_true",
        help="Force sequence-driven PFM reconstruction before direct motif comparison.",
    )
    motif_group.add_argument(
        "--pfm-top-fraction",
        type=float,
        default=0.05,
        help="Fraction of top-scoring reconstructed sites used for cross-type PFM comparison. (default: %(default)s)",
    )

    _add_common_technical_arguments(parser)
    _add_significance_arguments(parser)


def _add_build_null_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the null-distribution builder parser."""
    parser = subparsers.add_parser(
        "build-null",
        help="Build a pooled null distribution from unrelated target motif comparisons.",
    )
    parser.add_argument("motifs", help="Motif collection: directory or multi-motif MEME file.")
    parser.add_argument("--model-type", choices=MOTIF_MODEL_TYPES, required=True, help="Motif model format.")
    parser.add_argument("--pattern", help="Glob pattern used when loading a directory collection.")

    relation = parser.add_argument_group("Relation Options")
    relation.add_argument("--groups", required=True, help="TSV/CSV with motif and group columns.")
    relation.add_argument("--name-column", default="motif", help="Motif-name column for --groups.")
    relation.add_argument("--group-column", default="group", help="Group column for --groups.")
    relation.add_argument("--ignore-missing-relations", action="store_true", help="Ignore relation names not loaded.")

    comparison = parser.add_argument_group("Comparison Options")
    comparison.add_argument("--strategy", choices=["profile", "motif"], required=True)
    comparison.add_argument("--metric", help="Comparison metric. Defaults to co for profile and pcc for motif.")
    comparison.add_argument("--fasta", help="FASTA sequences used by profile mode or sequence-driven PFM mode.")
    comparison.add_argument("--background", help="Optional FASTA background sequences for profile normalization.")
    comparison.add_argument("--num-sequences", type=int, default=1000)
    comparison.add_argument("--seq-length", type=int, default=200)
    comparison.add_argument("--search-range", type=int, default=10)
    comparison.add_argument("--window-radius", type=int, default=10)
    comparison.add_argument("--realign-window", type=int, default=3)
    comparison.add_argument("--min-logfpr", type=float, default=None)
    comparison.add_argument("--pfm-mode", action="store_true")
    comparison.add_argument("--pfm-top-fraction", type=float, default=0.05)
    comparison.add_argument("--cache", choices=["off", "on"], default="off")
    comparison.add_argument("--cache-dir", default=".mimosa-cache")

    output = parser.add_argument_group("Output Options")
    output.add_argument("--output", required=True, help="Path to write the trusted joblib null distribution file.")
    output.add_argument(
        "--install-cache",
        action="store_true",
        help="Also copy the null distribution file into the user cache.",
    )
    output.add_argument("--strict", action="store_true", help="Fail when a query lacks enough null targets.")
    output.add_argument("--min-null-targets", type=int, default=1)

    _add_common_technical_arguments(parser)


def _add_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the cache management parser."""
    parser = subparsers.add_parser("cache", help="Manage lazy profile cache artifacts.")
    nested = parser.add_subparsers(dest="cache_action", required=True)

    clear_parser = nested.add_parser("clear", help="Remove all cached profile artifacts.")
    clear_parser.add_argument(
        "--cache-dir",
        default=".mimosa-cache",
        help="Directory containing cached profile artifacts. (default: %(default)s)",
    )
    clear_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def validate_inputs(args) -> None:
    """Validate input files and parameters."""
    if args.mode == "cache":
        return

    logger = logging.getLogger(__name__)
    try:
        if args.mode == "build-null":
            _validate_build_null_inputs(args)
        else:
            _validate_comparison_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


def _validate_build_null_inputs(args) -> None:
    """Validate files and parameters for null-distribution building."""
    file_checks = [(args.motifs, "Motif collection")]
    for optional_path, label in (
        (getattr(args, "groups", None), "Group relation file"),
        (getattr(args, "fasta", None), "FASTA file"),
        (getattr(args, "background", None), "Background FASTA file"),
    ):
        if optional_path:
            file_checks.append((optional_path, label))

    _validate_existing_paths(file_checks)
    create_comparator_config(**map_args_to_comparator_kwargs(args))
    _validate_build_null_metric(args)
    validate_positive_int("min_null_targets", args.min_null_targets)


def _validate_comparison_inputs(args) -> None:
    """Validate files and comparator parameters for comparison modes."""
    file_checks = [
        (args.model1, "Input file"),
        (args.model2, "Input file"),
    ]
    if getattr(args, "fasta", None):
        file_checks.append((args.fasta, "FASTA file"))
    if getattr(args, "background", None):
        file_checks.append((args.background, "Background FASTA file"))
    if getattr(args, "null_distribution", None):
        file_checks.append((args.null_distribution, "Null distribution file"))

    _validate_existing_paths(file_checks)
    if args.mode in {"profile", "motif"}:
        create_comparator_config(**map_args_to_comparator_kwargs(args))


def _validate_existing_paths(file_checks) -> None:
    """Validate a sequence of path/label pairs."""
    for path, label in file_checks:
        validate_file_exists(path, label)


def map_args_to_comparator_kwargs(args) -> Dict[str, Any]:
    """Map CLI arguments to comparator configuration kwargs."""
    if args.mode == "profile":
        return {
            "metric": args.metric,
            "n_jobs": args.jobs,
            "seed": args.seed,
            "search_range": args.search_range,
            "window_radius": args.window_radius,
            "realign_window": args.realign_window,
            "min_logfpr": args.min_logfpr,
            "cache_mode": args.cache,
            "cache_dir": args.cache_dir,
            "pvalue": args.pvalue,
            "null_distribution": args.null_distribution,
            "null_search_dirs": args.null_search_dirs,
            "effective_number_of_targets": args.effective_number_of_targets,
        }

    if args.mode == "motif":
        return {
            "metric": args.metric,
            "n_jobs": args.jobs,
            "seed": args.seed,
            "pfm_mode": args.pfm_mode,
            "pfm_top_fraction": args.pfm_top_fraction,
            "pvalue": args.pvalue,
            "null_distribution": args.null_distribution,
            "null_search_dirs": args.null_search_dirs,
            "effective_number_of_targets": args.effective_number_of_targets,
        }

    if args.mode == "build-null":
        metric = args.metric or ("co" if args.strategy == "profile" else "pcc")
        return {
            "metric": validate_metric(metric),
            "n_jobs": args.jobs,
            "seed": args.seed,
            "search_range": args.search_range,
            "window_radius": args.window_radius,
            "realign_window": args.realign_window,
            "min_logfpr": args.min_logfpr,
            "pfm_mode": args.pfm_mode,
            "pfm_top_fraction": args.pfm_top_fraction,
            "cache_mode": args.cache,
            "cache_dir": args.cache_dir,
            "pvalue": False,
        }

    return {}


def _validate_build_null_metric(args) -> None:
    metric = args.metric or ("co" if args.strategy == "profile" else "pcc")
    allowed = SUPPORTED_PROFILE_METRICS if args.strategy == "profile" else SUPPORTED_MOTIF_METRICS
    if metric not in allowed:
        raise ValueError(f"Strategy '{args.strategy}' requires one of: {', '.join(allowed)}")


def build_comparison_config_from_args(args):
    """Build OneToOneConfig from parsed CLI args."""
    comparator_kwargs = map_args_to_comparator_kwargs(args)
    comparator = create_comparator_config(**comparator_kwargs)

    sequences = getattr(args, "fasta", None)
    background = getattr(args, "background", None)

    return create_one_to_one_config(
        query=args.model1,
        target=args.model2,
        query_type=args.model1_type,
        target_type=args.model2_type,
        strategy=args.mode,
        sequences=sequences,
        background=background,
        num_sequences=getattr(args, "num_sequences", 1000),
        seq_length=getattr(args, "seq_length", 200),
        seed=getattr(args, "seed", 127),
        comparator=comparator,
    )


def run_comparison_from_args(args) -> None:
    """Run comparison with parsed CLI arguments."""
    logger = logging.getLogger(__name__)
    logger.info("Running comparison in mode: %s", args.mode)

    try:
        config = build_comparison_config_from_args(args)
        result = run_one_to_one(config)
        logger.info("Comparison completed successfully")
        print(json.dumps(result.to_dict()))
    except Exception as exc:
        logger.error("Comparison execution failed: %s", exc)
        raise


def run_cache_command_from_args(args) -> None:
    """Run a cache maintenance command."""
    logger = logging.getLogger(__name__)

    if args.cache_action == "clear":
        removed = clear_cache(args.cache_dir)
        logger.info("Cleared cache directory '%s' (%s entries removed).", args.cache_dir, removed)
        print(json.dumps({"cache_dir": args.cache_dir, "removed": removed}))
        return

    raise ValueError(f"Unknown cache action: {args.cache_action}")


def build_null_request_from_args(args) -> NullBuildRequest:
    """Resolve parsed CLI arguments into a null-distribution build request."""
    models = read_models(args.motifs, args.model_type, pattern=args.pattern)
    known_names = {model.name for model in models}
    relation_fingerprint = file_fingerprint(args.groups)
    relations = parse_group_relations(
        args.groups,
        name_column=args.name_column,
        group_column=args.group_column,
        ignore_missing=args.ignore_missing_relations,
        known_names=known_names,
    )

    comparator = create_comparator_config(**map_args_to_comparator_kwargs(args))
    sequences = _resolve_build_null_sequences(args, comparator)
    background = read_fasta(args.background) if args.background else None
    return NullBuildRequest(
        models=models,
        relations=relations,
        strategy=args.strategy,
        config=comparator,
        output=args.output,
        sequences=sequences,
        background=background,
        min_null_targets=args.min_null_targets,
        strict=args.strict,
        relation_fingerprint=relation_fingerprint,
        install_cache=args.install_cache,
        progress=getattr(args, "progress", False),
    )


def run_build_null_from_args(args) -> None:
    """Build and save a null distribution file from CLI arguments."""
    request = build_null_request_from_args(args)
    summary = run_build_null_request(request)
    print(json.dumps(summary.to_dict()))


def _resolve_build_null_sequences(args, comparator):
    """Resolve sequences for null building when the selected score path needs them."""
    needs_sequences = args.strategy == "profile" or bool(comparator["pfm_mode"])
    if not needs_sequences:
        return None
    if args.fasta:
        return read_fasta(args.fasta)

    num_sequences = validate_positive_int("num_sequences", args.num_sequences)
    seq_length = validate_positive_int("seq_length", args.seq_length)
    return make_random_sequence_batch(num_sequences, seq_length, args.seed)


def main_cli() -> None:
    """Main CLI entry point."""
    parser = create_arg_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    setup_logging(args.verbose, getattr(args, "progress", False))
    validate_inputs(args)
    if args.mode == "cache":
        run_cache_command_from_args(args)
    elif args.mode == "build-null":
        run_build_null_from_args(args)
    else:
        run_comparison_from_args(args)


if __name__ == "__main__":
    main_cli()
