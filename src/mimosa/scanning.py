"""Scan dispatch and score calibration helpers for motif models."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from mimosa.batches import (
    SCORE_PADDING,
    flatten_profile_bundle,
    flatten_valid,
    make_strand_bundle,
    pack_batch,
)
from mimosa.functions import (
    batch_all_scores,
    batch_all_scores_strands,
    build_score_log_tail_table,
    scores_to_empirical_log_tail,
    scores_to_empirical_log_tail_bundle,
)
from mimosa.models import GenericModel, get_model_handler

StrandMode = Literal["best", "+", "-", "both"]
_STRAND_MODES = frozenset(("best", "+", "-", "both"))


def resolve_strand_mode(strand: Optional[StrandMode], default: StrandMode = "best") -> StrandMode:
    """Normalize and validate one public strand mode."""
    normalized = str(strand or default).lower()
    if normalized not in _STRAND_MODES:
        available = ", ".join(sorted(_STRAND_MODES))
        raise ValueError(f"strand must be one of: {available}")
    return normalized  # type: ignore[return-value]


def scan_model(model: GenericModel, sequences=None, strand: Optional[StrandMode] = None):
    """Universal scanning function that dispatches to the appropriate handler."""
    handler = get_model_handler(model.type_key)
    strand_mode = resolve_strand_mode(strand, model.config.get("strand_mode", "best"))
    if strand_mode == "both":
        return scan_model_strands(model, sequences)
    return handler.scan(model, sequences, strand_mode)


def scan_model_strands(model: GenericModel, sequences=None):
    """Scan one model on both strands, using a shared backend call when available."""
    handler = get_model_handler(model.type_key)
    scan_both = handler.scan_both
    if scan_both is not None:
        plus_scores, minus_scores = scan_both(model, sequences)
    else:
        plus_scores = handler.scan(model, sequences, "+")
        minus_scores = handler.scan(model, sequences, "-")
    return make_strand_bundle(plus_scores, minus_scores)


def get_score_bounds(model: GenericModel) -> tuple[float, float]:
    """Return theoretical minimum and maximum scores for a model."""
    return get_model_handler(model.type_key).score_bounds(model)


def score_bounds_from_representation(representation: np.ndarray) -> tuple[float, float]:
    """Compute theoretical score bounds from one model tensor."""
    minimum = representation.min(axis=tuple(range(representation.ndim - 1))).sum()
    maximum = representation.max(axis=tuple(range(representation.ndim - 1))).sum()
    return minimum, maximum


def calculate_threshold_table(model: GenericModel, sequences, strand: StrandMode = "both") -> np.ndarray:
    """Calculate a score-to-log-tail lookup table on explicitly provided sequences."""
    scores = scan_model(model, sequences, strand=resolve_strand_mode(strand))
    return build_score_log_tail_table(flatten_scan_scores(scores)).astype(np.float64, copy=False)


def get_frequencies(model: GenericModel, sequences, strand: Optional[StrandMode] = None):
    """Calculate per-position empirical log-tail values."""
    scores = scan_model(model, sequences, strand)
    if "mask" in scores:
        return scores_to_empirical_log_tail(scores)
    return scores_to_empirical_log_tail_bundle(scores)


def get_scores(model: GenericModel, sequences, strand: Optional[StrandMode] = None):
    """Calculate motif scores for each position."""
    return scan_model(model, sequences, strand)


def flatten_scan_scores(scores) -> np.ndarray:
    """Flatten either a masked score batch or a strand-aware profile bundle."""
    if "mask" in scores:
        return flatten_valid(scores)
    return flatten_profile_bundle(scores)


def scan_with_batch_kernel(model: GenericModel, sequences, strand: StrandMode, *, with_context: bool = False):
    """Scan a tensor-based motif model with the shared Numba batch kernel."""
    representation = np.asarray(model.representation, dtype=np.float32)
    kmer = int(model.config.get("kmer", 1))

    if strand == "+":
        return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=with_context)
    if strand == "-":
        return batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=with_context)
    if strand == "best":
        sf, sr = batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)
        values = np.full(sf["values"].shape, SCORE_PADDING, dtype=np.float32)
        values[sf["mask"]] = np.maximum(sf["values"][sf["mask"]], sr["values"][sr["mask"]])
        return pack_batch(values, sf["mask"], sf["lengths"], SCORE_PADDING)
    if strand == "both":
        sf, sr = batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)
        return make_strand_bundle(sf, sr)
    raise ValueError(f"Invalid strand mode: {strand}")


def scan_with_batch_kernel_strands(model: GenericModel, sequences, *, with_context: bool = False):
    """Scan a tensor-based motif model on both strands in one shared Numba call."""
    representation = np.asarray(model.representation, dtype=np.float32)
    kmer = int(model.config.get("kmer", 1))
    return batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)


def score_bounds_from_model(model: GenericModel) -> tuple[float, float]:
    """Return theoretical min/max score for tensor-based motif models."""
    return score_bounds_from_representation(np.asarray(model.representation))


__all__ = [
    "StrandMode",
    "calculate_threshold_table",
    "flatten_scan_scores",
    "get_frequencies",
    "get_score_bounds",
    "get_scores",
    "resolve_strand_mode",
    "scan_model",
    "scan_model_strands",
    "score_bounds_from_model",
    "score_bounds_from_representation",
    "scan_with_batch_kernel",
    "scan_with_batch_kernel_strands",
]
