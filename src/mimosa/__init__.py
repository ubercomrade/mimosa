"""Mimosa: motif scanning, comparison, and statistical evaluation in Python."""

# Imports below intentionally follow the CLI thread-budget bootstrap.
# ruff: noqa: E402

import os
import sys
from importlib.metadata import version


def _configure_cli_numba_threads():
    if len(sys.argv) < 2 or sys.argv[1] not in {"compare", "compare-many"}:
        return
    for index, argument in enumerate(sys.argv[2:], start=2):
        if argument.startswith("--numba-threads="):
            value = argument.split("=", 1)[1]
        elif argument == "--numba-threads" and index + 1 < len(sys.argv):
            value = sys.argv[index + 1]
        else:
            continue
        try:
            value = int(value)
        except ValueError:
            return
        if 1 <= value <= 4:
            os.environ["NUMBA_NUM_THREADS"] = str(value)
        return


_configure_cli_numba_threads()

from .arrays import (
    EncodedSequences,
    N_CODE,
    RaggedArray,
    StrandPair,
    encode_sequence,
    reverse_complement,
)
from .compare import ComparisonResult, compare, compare_many
from .errors import (
    InvariantError,
    MimosaError,
    ModelDimensionError,
    ModelFormatError,
    ModelInterfaceError,
)
from .models import (
    BaMM,
    Dimont,
    MotifModel,
    PWM,
    SiteGA,
    Slim,
    extend_pwm_with_n,
    pfm_to_pwm,
    pwm_from_pfm,
    site_start_offset,
    window_size,
)
from .profiles.alignment import (
    ProfileConfig,
    parse_profile_metric,
)
from .profiles.normalization import (
    EmpiricalLogTail,
    HybridEmpiricalLogTail,
    normalization_fingerprint,
)
from .profiles.prepared import PreparedProfile, ScoreProfile, prepare_profile
from .scan import scan
from .sites import (
    BestPerSequence,
    SiteCollection,
    ThresholdHits,
    TopFractionHits,
    build_pcm,
    extract_site_matrix,
    pcm_to_pfm,
    reconstruct_pfm,
    select_sites,
    site_strings,
)
from .statistics import (
    AnnotatedResult,
    NullDistribution,
    adjusted_pvalues,
    annotate_results,
    build_null,
    empirical_upper_tail_pvalue,
    evalue,
)
from .io import read_fasta, read_model, read_scores, write_model

__version__ = version("mimosa-tool")

__all__ = [
    "EncodedSequences",
    "N_CODE",
    "RaggedArray",
    "StrandPair",
    "encode_sequence",
    "reverse_complement",
    "ComparisonResult",
    "compare",
    "compare_many",
    "InvariantError",
    "MimosaError",
    "ModelDimensionError",
    "ModelFormatError",
    "ModelInterfaceError",
    "BaMM",
    "Dimont",
    "MotifModel",
    "PWM",
    "SiteGA",
    "Slim",
    "extend_pwm_with_n",
    "pfm_to_pwm",
    "pwm_from_pfm",
    "site_start_offset",
    "window_size",
    "ProfileConfig",
    "parse_profile_metric",
    "EmpiricalLogTail",
    "HybridEmpiricalLogTail",
    "normalization_fingerprint",
    "PreparedProfile",
    "ScoreProfile",
    "prepare_profile",
    "scan",
    "BestPerSequence",
    "SiteCollection",
    "ThresholdHits",
    "TopFractionHits",
    "build_pcm",
    "extract_site_matrix",
    "pcm_to_pfm",
    "reconstruct_pfm",
    "select_sites",
    "site_strings",
    "AnnotatedResult",
    "NullDistribution",
    "adjusted_pvalues",
    "annotate_results",
    "build_null",
    "empirical_upper_tail_pvalue",
    "evalue",
    "read_model",
    "write_model",
    "read_fasta",
    "read_scores",
]
