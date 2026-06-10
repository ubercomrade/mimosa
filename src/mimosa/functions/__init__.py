"""Numerical helpers and Numba-backed scoring kernels."""

from __future__ import annotations

from mimosa.functions.curves import cut_prc, cut_roc, precision_recall_curve, roc_curve, standardized_pauc
from mimosa.functions.formatting import format_params
from mimosa.functions.matrices import pcm_to_pfm, pfm_to_pwm
from mimosa.functions.profile import (
    calc_co,
    calc_dice,
    prepare_profile_bundle,
    rowwise_co,
    rowwise_cosine,
    rowwise_dice,
)
from mimosa.functions.scanning import batch_all_scores, batch_all_scores_strands, score_seq
from mimosa.functions.tails import (
    apply_score_log_tail_table,
    apply_score_log_tail_table_to_profile_bundle,
    build_score_log_tail_table,
    lookup_score_for_tail_probability,
    normalize_empirical_log_tail_pair,
    scores_to_empirical_log_tail,
    scores_to_empirical_log_tail_bundle,
)

__all__ = [
    "apply_score_log_tail_table",
    "apply_score_log_tail_table_to_profile_bundle",
    "batch_all_scores",
    "batch_all_scores_strands",
    "build_score_log_tail_table",
    "calc_co",
    "calc_dice",
    "cut_prc",
    "cut_roc",
    "format_params",
    "lookup_score_for_tail_probability",
    "normalize_empirical_log_tail_pair",
    "pcm_to_pfm",
    "pfm_to_pwm",
    "precision_recall_curve",
    "prepare_profile_bundle",
    "roc_curve",
    "rowwise_co",
    "rowwise_cosine",
    "rowwise_dice",
    "score_seq",
    "scores_to_empirical_log_tail",
    "scores_to_empirical_log_tail_bundle",
    "standardized_pauc",
]
