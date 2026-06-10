"""Motif hit extraction, site tables, and PFM reconstruction."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from mimosa.batches import (
    make_strand_bundle,
    pack_batch,
    profile_row_values,
    row_values,
)
from mimosa.functions import lookup_score_for_tail_probability, pcm_to_pfm
from mimosa.models import GenericModel
from mimosa.scanning import StrandMode, calculate_threshold_table, resolve_strand_mode, scan_model, scan_model_strands
from mimosa.validation import validate_site_mode

_SEQ_DECODER = np.array(["A", "C", "G", "T", "N"], dtype="U1")
_NUCLEOTIDE_CARDINALITY = 4


def _empty_hit_arrays() -> dict[str, np.ndarray]:
    """Return a consistent empty hit-array payload."""
    return {
        "seq_index": np.empty(0, dtype=np.int64),
        "start": np.empty(0, dtype=np.int64),
        "strand_idx": np.empty(0, dtype=np.int8),
        "score": np.empty(0, dtype=np.float32),
    }


def _strand_indices_for_hit_mode(strand: StrandMode) -> tuple[int, ...]:
    """Return strand indices considered by one hit-collection request."""
    plus_strand = 0
    minus_strand = 1
    if strand == "+":
        return (plus_strand,)
    if strand == "-":
        return (minus_strand,)
    return (plus_strand, minus_strand)


def _empty_score_batch_like(score_batch):
    """Return an empty score batch with the same shape and row lengths."""
    score_values = np.asarray(score_batch["values"])
    padding_value = score_batch["padding_value"]
    values = np.full(score_values.shape, padding_value, dtype=score_values.dtype)
    mask = np.zeros(np.asarray(score_batch["mask"]).shape, dtype=bool)
    return pack_batch(values, mask, score_batch["lengths"], padding_value)


def _scan_bundle_for_strand(model: GenericModel, sequences, strand: StrandMode):
    """Scan the minimum required strand set and expose it as a two-strand bundle."""
    if strand in {"best", "both"}:
        return scan_model_strands(model, sequences)

    scores = scan_model(model, sequences, strand=strand)
    empty_scores = _empty_score_batch_like(scores)
    if strand == "+":
        return make_strand_bundle(scores, empty_scores)
    return make_strand_bundle(empty_scores, scores)


def _collect_best_hits(score_bundle, strand: StrandMode) -> dict[str, np.ndarray]:
    """Collect the single best hit per sequence as numeric arrays."""
    seq_indices: list[int] = []
    starts: list[int] = []
    strand_indices: list[int] = []
    scores: list[float] = []
    candidate_strands = _strand_indices_for_hit_mode(strand)
    plus_strand = 0

    for seq_idx in range(len(score_bundle["lengths"])):
        best_score = -np.inf
        best_start = -1
        best_strand = plus_strand

        for strand_idx in candidate_strands:
            strand_scores = profile_row_values(score_bundle, strand_idx, seq_idx)
            if strand_scores.size == 0:
                continue
            strand_start = int(np.argmax(strand_scores))
            strand_score = float(strand_scores[strand_start])
            if strand_score > best_score:
                best_score = strand_score
                best_start = strand_start
                best_strand = int(strand_idx)

        if best_start < 0 or not np.isfinite(best_score):
            continue
        seq_indices.append(seq_idx)
        starts.append(best_start)
        strand_indices.append(best_strand)
        scores.append(best_score)

    if not seq_indices:
        return _empty_hit_arrays()

    return {
        "seq_index": np.asarray(seq_indices, dtype=np.int64),
        "start": np.asarray(starts, dtype=np.int64),
        "strand_idx": np.asarray(strand_indices, dtype=np.int8),
        "score": np.asarray(scores, dtype=np.float32),
    }


def _collect_threshold_hits_for_strands(
    score_bundle, score_threshold: float, strand_indices: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Collect all strand-specific hits above threshold as numeric arrays."""
    seq_indices_parts = []
    start_parts = []
    strand_parts = []
    score_parts = []

    for seq_idx in range(len(score_bundle["lengths"])):
        for strand_idx in strand_indices:
            strand_scores = profile_row_values(score_bundle, strand_idx, seq_idx)
            strand_positions = np.flatnonzero(strand_scores >= score_threshold)
            if strand_positions.size == 0:
                continue
            seq_indices_parts.append(np.full(strand_positions.size, seq_idx, dtype=np.int64))
            start_parts.append(strand_positions.astype(np.int64, copy=False))
            strand_parts.append(np.full(strand_positions.size, strand_idx, dtype=np.int8))
            score_parts.append(strand_scores[strand_positions].astype(np.float32, copy=False))

    if not seq_indices_parts:
        return _empty_hit_arrays()

    return {
        "seq_index": np.concatenate(seq_indices_parts),
        "start": np.concatenate(start_parts),
        "strand_idx": np.concatenate(strand_parts),
        "score": np.concatenate(score_parts),
    }


def _collect_best_strand_threshold_hits(score_bundle, score_threshold: float) -> dict[str, np.ndarray]:
    """Collect above-threshold hits after collapsing both strands by per-position maximum."""
    plus_strand = 0
    minus_strand = 1
    seq_indices_parts = []
    start_parts = []
    strand_parts = []
    score_parts = []

    for seq_idx in range(len(score_bundle["lengths"])):
        plus_scores = profile_row_values(score_bundle, plus_strand, seq_idx)
        minus_scores = profile_row_values(score_bundle, minus_strand, seq_idx)
        if plus_scores.size == 0:
            continue

        best_scores = np.maximum(plus_scores, minus_scores)
        positions = np.flatnonzero(best_scores >= score_threshold)
        if positions.size == 0:
            continue

        seq_indices_parts.append(np.full(positions.size, seq_idx, dtype=np.int64))
        start_parts.append(positions.astype(np.int64, copy=False))
        strand_parts.append(
            np.where(plus_scores[positions] >= minus_scores[positions], plus_strand, minus_strand).astype(
                np.int8,
                copy=False,
            )
        )
        score_parts.append(best_scores[positions].astype(np.float32, copy=False))

    if not seq_indices_parts:
        return _empty_hit_arrays()

    return {
        "seq_index": np.concatenate(seq_indices_parts),
        "start": np.concatenate(start_parts),
        "strand_idx": np.concatenate(strand_parts),
        "score": np.concatenate(score_parts),
    }


def _collect_threshold_hits(score_bundle, score_threshold: float, strand: StrandMode) -> dict[str, np.ndarray]:
    """Collect threshold hits using the requested strand semantics."""
    if strand == "best":
        return _collect_best_strand_threshold_hits(score_bundle, score_threshold)
    return _collect_threshold_hits_for_strands(score_bundle, score_threshold, _strand_indices_for_hit_mode(strand))


def _collect_hits(
    model: GenericModel, sequences, mode: str, score_threshold: Optional[float], strand: StrandMode
) -> dict[str, np.ndarray]:
    """Collect motif hits as numeric arrays."""
    score_bundle = _scan_bundle_for_strand(model, sequences, strand)
    if mode == "best":
        return _collect_best_hits(score_bundle, strand)
    if score_threshold is None:
        raise ValueError("score_threshold is required in threshold mode")
    return _collect_threshold_hits(score_bundle, float(score_threshold), strand)


def _scores_to_log_tail_array(scores: np.ndarray, threshold_table: np.ndarray) -> np.ndarray:
    """Convert one score array to log-tail values using an explicit lookup table."""
    if scores.size == 0:
        return np.empty(0, dtype=np.float64)

    scores_col = threshold_table[:, 0]
    log_tail_col = threshold_table[:, 1]
    idx = np.searchsorted(-scores_col, -scores.astype(np.float64, copy=False), side="left")
    idx = np.clip(idx, 0, len(log_tail_col) - 1)
    return log_tail_col[idx]


def _resolve_hit_threshold_table(
    model: GenericModel, sequences, background_sequences, threshold_table, strand: StrandMode
) -> np.ndarray:
    """Resolve the explicit log-tail table used for hit extraction and annotation."""
    if threshold_table is not None:
        return np.asarray(threshold_table, dtype=np.float64)

    calibration_sequences = background_sequences if background_sequences is not None else sequences
    return calculate_threshold_table(model, calibration_sequences, strand=strand)


def _resolve_hits(model: GenericModel, sequences, selection: dict, *, include_threshold_table: bool) -> dict:
    """Resolve hit arrays and optional threshold metadata for one request."""
    mode = validate_site_mode(selection.get("mode", "best"), selection.get("fpr_threshold"))
    strand = resolve_strand_mode(selection.get("strand"), "both")
    threshold_table = None
    score_threshold = None

    if include_threshold_table or mode == "threshold":
        threshold_table = _resolve_hit_threshold_table(
            model,
            sequences,
            selection.get("background_sequences"),
            selection.get("threshold_table"),
            strand,
        )

    if mode == "threshold":
        if threshold_table is None:
            raise ValueError("threshold_table is required in threshold mode")
        score_threshold = lookup_score_for_tail_probability(threshold_table, float(selection["fpr_threshold"]))
        logging.getLogger(__name__).info(
            "FPR threshold: %s -> score threshold: %.4f", selection["fpr_threshold"], score_threshold
        )

    hit_arrays = _sort_hit_arrays(_collect_hits(model, sequences, mode, score_threshold, strand))
    return {
        "hit_arrays": hit_arrays,
        "threshold_table": threshold_table,
        "score_threshold": score_threshold,
        "strand": strand,
    }


def _extract_site_matrix(
    sequences, seq_indices: np.ndarray, starts: np.ndarray, motif_length: int, strand_indices=None
):
    """Extract numeric motif windows for a set of hits."""
    n_hits = seq_indices.size
    sites = np.empty((n_hits, motif_length), dtype=sequences["values"].dtype)

    for hit_idx in range(n_hits):
        seq = row_values(sequences, int(seq_indices[hit_idx]))
        start = int(starts[hit_idx])
        sites[hit_idx] = seq[start : start + motif_length]

    if strand_indices is not None:
        minus_mask = strand_indices == 1
        if np.any(minus_mask):
            reversed_sites = sites[minus_mask, ::-1]
            padding_value = _NUCLEOTIDE_CARDINALITY
            complement_offset = _NUCLEOTIDE_CARDINALITY - 1
            sites[minus_mask] = np.where(
                reversed_sites == padding_value,
                padding_value,
                complement_offset - reversed_sites,
            )

    return sites


def _site_matrix_to_strings(site_matrix: np.ndarray) -> np.ndarray:
    """Convert numeric site windows into DNA strings."""
    if site_matrix.size == 0:
        return np.empty(0, dtype=object)

    decoded = _SEQ_DECODER[np.clip(site_matrix, 0, 4)]
    return np.fromiter(("".join(row) for row in decoded), dtype=object, count=decoded.shape[0])


def _build_pcm_from_site_matrix(site_matrix: np.ndarray, motif_length: int) -> np.ndarray:
    """Build a position count matrix from numeric sites."""
    pcm = np.zeros((4, motif_length), dtype=np.float32)
    if site_matrix.size == 0:
        return pcm

    valid_mask = site_matrix < _NUCLEOTIDE_CARDINALITY
    col_idx = np.broadcast_to(np.arange(motif_length, dtype=np.int64), site_matrix.shape)
    np.add.at(pcm, (site_matrix[valid_mask], col_idx[valid_mask]), 1.0)
    return pcm


def _sort_hit_arrays(hit_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort hits by sequence index ascending and score descending."""
    if hit_arrays["score"].size == 0:
        return hit_arrays
    order = np.lexsort(
        (
            hit_arrays["strand_idx"],
            hit_arrays["start"],
            -hit_arrays["score"],
            hit_arrays["seq_index"],
        )
    )
    return {key: values[order] for key, values in hit_arrays.items()}


def _select_top_hit_arrays(hit_arrays: dict[str, np.ndarray], top_fraction: Optional[float]) -> dict[str, np.ndarray]:
    """Keep only the top-scoring hits."""
    if top_fraction is None or hit_arrays["score"].size == 0:
        return hit_arrays

    n_hits = hit_arrays["score"].size
    n_keep = max(1, int(n_hits * top_fraction))
    if n_keep >= n_hits:
        return hit_arrays

    keep_idx = np.argpartition(hit_arrays["score"], n_hits - n_keep)[-n_keep:]
    keep_idx = keep_idx[np.argsort(hit_arrays["score"][keep_idx])[::-1]]
    return {key: values[keep_idx] for key, values in hit_arrays.items()}


def _empty_sites_frame() -> pd.DataFrame:
    """Return an empty site table with the public schema."""
    return pd.DataFrame(
        {
            "seq_index": np.empty(0, dtype=np.int64),
            "start": np.empty(0, dtype=np.int64),
            "end": np.empty(0, dtype=np.int64),
            "strand": np.empty(0, dtype=object),
            "score": np.empty(0, dtype=np.float32),
            "log_tail": np.empty(0, dtype=np.float64),
            "site": np.empty(0, dtype=object),
        }
    )


def _build_sites_frame(
    model: GenericModel, sequences, hit_arrays: dict[str, np.ndarray], threshold_table: np.ndarray
) -> pd.DataFrame:
    """Build the public site table from resolved hits."""
    if threshold_table is None:
        raise ValueError("threshold_table is required to annotate sites with log-tail values")
    if hit_arrays["score"].size == 0:
        return _empty_sites_frame()

    site_matrix = _extract_site_matrix(
        sequences,
        hit_arrays["seq_index"],
        hit_arrays["start"],
        model.length,
        hit_arrays["strand_idx"],
    )
    return pd.DataFrame(
        {
            "seq_index": hit_arrays["seq_index"],
            "start": hit_arrays["start"],
            "end": hit_arrays["start"] + model.length,
            "strand": np.where(hit_arrays["strand_idx"] == 0, "+", "-"),
            "score": hit_arrays["score"],
            "log_tail": _scores_to_log_tail_array(hit_arrays["score"], threshold_table),
            "site": _site_matrix_to_strings(site_matrix),
        }
    )


def _hits_to_pfm(model: GenericModel, sequences, hit_arrays: dict[str, np.ndarray], pseudocount: float) -> np.ndarray:
    """Convert a selected hit set to a PFM."""
    if hit_arrays["score"].size == 0:
        raise ValueError("No sites found")

    site_matrix = _extract_site_matrix(
        sequences,
        hit_arrays["seq_index"],
        hit_arrays["start"],
        model.length,
        hit_arrays["strand_idx"],
    )
    pcm = _build_pcm_from_site_matrix(site_matrix, model.length)
    return pcm_to_pfm(pcm, pseudocount=pseudocount).astype(np.float32, copy=False)


def get_sites(
    model: GenericModel,
    sequences,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    strand: StrandMode = "both",
    background_sequences=None,
    threshold_table: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Find motif binding sites in sequences."""
    resolved = _resolve_hits(
        model,
        sequences,
        {
            "mode": mode,
            "fpr_threshold": fpr_threshold,
            "strand": strand,
            "background_sequences": background_sequences,
            "threshold_table": threshold_table,
        },
        include_threshold_table=True,
    )
    df = _build_sites_frame(model, sequences, resolved["hit_arrays"], resolved["threshold_table"])
    logging.getLogger(__name__).info("Found %s site(s) in %s sequence(s)", len(df), len(sequences["lengths"]))
    return df


def get_pfm(
    model: GenericModel,
    sequences,
    mode: str = "best",
    fpr_threshold: Optional[float] = None,
    strand: StrandMode = "both",
    background_sequences=None,
    threshold_table: Optional[np.ndarray] = None,
    top_fraction: Optional[float] = None,
    pseudocount: float = 0.25,
) -> np.ndarray:
    """Construct a Position Frequency Matrix from binding sites."""
    logger = logging.getLogger(__name__)
    logger.info("Computing PFM for model: %s", model.name)
    resolved = _resolve_hits(
        model,
        sequences,
        {
            "mode": mode,
            "fpr_threshold": fpr_threshold,
            "strand": strand,
            "background_sequences": background_sequences,
            "threshold_table": threshold_table,
        },
        include_threshold_table=mode == "threshold",
    )
    selected_hits = _select_top_hit_arrays(resolved["hit_arrays"], top_fraction)
    if top_fraction is not None:
        logger.info("Selected top %.1f%%: %s sites", top_fraction * 100.0, selected_hits["score"].size)
    return _hits_to_pfm(model, sequences, selected_hits, pseudocount)


__all__ = ["get_pfm", "get_sites"]
