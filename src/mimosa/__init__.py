"""Mimosa: motif scanning, comparison, and statistical evaluation in Python."""

from importlib.metadata import version

from .arrays import (
    EncodedSequences,
    N_CODE,
    RaggedArray,
    StrandPair,
    encode_base,
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
    reverse_complement_weights,
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
    "encode_base",
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
    "reverse_complement_weights",
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
