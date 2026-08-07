"""Site selection, extraction, and PFM reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import N_CODE, RaggedArray, StrandPair, reverse_complement
from .errors import InvariantError, ModelDimensionError, ModelFormatError
from .models import site_start_offset
from .scan import _scan_batch_into

NUCLEOTIDE_CARDINALITY = 4


class SiteSelector:
    pass


class BestPerSequence(SiteSelector):
    pass


class ThresholdHits(SiteSelector):
    def __init__(self, threshold):
        self.threshold = np.float32(threshold)


class TopFractionHits(SiteSelector):
    def __init__(self, fraction, base=None):
        value = float(fraction)
        if not (np.isfinite(value) and 0.0 < value <= 1.0):
            raise ValueError("fraction must be finite and lie in (0, 1].")
        self.fraction = value
        self.base = base if base is not None else BestPerSequence()


@dataclass(frozen=True)
class SiteCollection:
    seq_indices: np.ndarray
    starts: np.ndarray
    strands: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        n = self.seq_indices.size
        if self.starts.size != n or self.strands.size != n or self.scores.size != n:
            raise ValueError("site collection arrays must have equal lengths.")
        if np.any(self.seq_indices < 0):
            raise ValueError("sequence indices must be non-negative.")
        if np.any(self.starts < 0):
            raise ValueError("site starts must be non-negative.")
        if np.any((self.strands != 0) & (self.strands != 1)):
            raise ValueError("strands must be 0 (forward) or 1 (reverse).")
        if not np.all(np.isfinite(self.scores)):
            raise ValueError("site scores must be finite.")

    def __len__(self):
        return self.seq_indices.size

    def to_dict(self):
        return {
            "seq_indices": self.seq_indices.tolist(),
            "starts": self.starts.tolist(),
            "strands": self.strands.tolist(),
            "scores": self.scores.tolist(),
        }


def _collect_best_hits(bundle):
    n = len(bundle.forward)
    seq_indices, starts, strands, scores = [], [], [], []
    for seq_idx in range(n):
        best_score = -np.inf
        best_start = -1
        best_strand = 0
        fwd = bundle.forward[seq_idx]
        if fwd.size > 0:
            fwd_start = int(np.argmax(fwd))
            fwd_score = fwd[fwd_start]
            if fwd_score > best_score:
                best_score = fwd_score
                best_start = fwd_start
                best_strand = 0
        rev = bundle.reverse[seq_idx]
        if rev.size > 0:
            rev_start = int(np.argmax(rev))
            rev_score = rev[rev_start]
            if rev_score > best_score:
                best_score = rev_score
                best_start = rev_start
                best_strand = 1
        if best_start < 0 or not np.isfinite(best_score):
            continue
        seq_indices.append(seq_idx)
        starts.append(best_start)
        strands.append(best_strand)
        scores.append(best_score)
    return SiteCollection(
        np.array(seq_indices, dtype=np.int64),
        np.array(starts, dtype=np.int64),
        np.array(strands, dtype=np.int8),
        np.array(scores, dtype=np.float32),
    )


def _collect_threshold_hits(bundle, threshold):
    seq_indices, starts, strands, scores = [], [], [], []
    for seq_idx in range(len(bundle.forward)):
        fwd = bundle.forward[seq_idx]
        for pos in range(fwd.size):
            if fwd[pos] >= threshold:
                seq_indices.append(seq_idx)
                starts.append(pos)
                strands.append(0)
                scores.append(fwd[pos])
        rev = bundle.reverse[seq_idx]
        for pos in range(rev.size):
            if rev[pos] >= threshold:
                seq_indices.append(seq_idx)
                starts.append(pos)
                strands.append(1)
                scores.append(rev[pos])
    return SiteCollection(
        np.array(seq_indices, dtype=np.int64),
        np.array(starts, dtype=np.int64),
        np.array(strands, dtype=np.int8),
        np.array(scores, dtype=np.float32),
    )


def _collect_best_strand_threshold_hits(bundle, threshold):
    seq_indices, starts, strands, scores = [], [], [], []
    for seq_idx in range(len(bundle.forward)):
        fwd = bundle.forward[seq_idx]
        rev = bundle.reverse[seq_idx]
        if fwd.size == 0:
            continue
        for pos in range(fwd.size):
            best = max(fwd[pos], rev[pos])
            if best >= threshold:
                seq_indices.append(seq_idx)
                starts.append(pos)
                strands.append(0 if fwd[pos] >= rev[pos] else 1)
                scores.append(best)
    return SiteCollection(
        np.array(seq_indices, dtype=np.int64),
        np.array(starts, dtype=np.int64),
        np.array(strands, dtype=np.int8),
        np.array(scores, dtype=np.float32),
    )


def _sort_hit_arrays(coll):
    n = len(coll)
    if n <= 1:
        return coll
    order = np.lexsort(
        (coll.strands, coll.starts, -coll.scores, coll.seq_indices)
    )
    return SiteCollection(
        coll.seq_indices[order],
        coll.starts[order],
        coll.strands[order],
        coll.scores[order],
    )


def _select_top_hit_arrays(coll, fraction):
    n = len(coll)
    if n == 0:
        return coll
    n_keep = max(1, int(np.floor(n * fraction)))
    if n_keep >= n:
        return coll
    keep = np.argsort(-coll.scores, kind="stable")[:n_keep]
    keep = keep[np.argsort(-coll.scores[keep], kind="stable")]
    return SiteCollection(
        coll.seq_indices[keep], coll.starts[keep], coll.strands[keep], coll.scores[keep]
    )


def _scan_bundle_for_sites(model, batch, strands):
    if strands == "forward":
        fwd = _scan_batch_into(model, batch, "forward")
        rev = RaggedArray(np.array([], dtype=np.float32), np.zeros(len(fwd) + 1, dtype=np.int64))
        return StrandPair(fwd, rev)
    if strands == "reverse":
        rev = _scan_batch_into(model, batch, "reverse")
        fwd = RaggedArray(np.array([], dtype=np.float32), np.zeros(len(rev) + 1, dtype=np.int64))
        return StrandPair(fwd, rev)
    return _scan_batch_into(model, batch, "both")


def _collect_hits_from_bundle(selector, bundle, strands):
    if isinstance(selector, BestPerSequence):
        return _collect_best_hits(bundle)
    if isinstance(selector, ThresholdHits):
        if strands == "best":
            return _collect_best_strand_threshold_hits(bundle, selector.threshold)
        return _collect_threshold_hits(bundle, selector.threshold)
    if isinstance(selector, TopFractionHits):
        base_coll = _collect_hits_from_bundle(selector.base, bundle, strands)
        return _select_top_hit_arrays(base_coll, selector.fraction)
    raise ValueError(f"unsupported selector: {selector!r}")


def select_sites(model, sequences, selector, *, strands="best"):
    if strands not in ("forward", "reverse", "best", "both"):
        raise ValueError(f"unsupported strand policy: {strands!r}")
    bundle = _scan_bundle_for_sites(model, sequences, strands)
    coll = _collect_hits_from_bundle(selector, bundle, strands)
    return _sort_hit_arrays(coll)


def extract_site_matrix(batch, coll, motif_width, site_offset=0):
    n_hits = len(coll)
    if motif_width <= 0:
        raise ValueError("motif_width must be positive.")
    if site_offset < 0:
        raise ValueError("site_offset must be non-negative.")
    if np.any(coll.seq_indices >= len(batch)):
        raise ValueError("site sequence indices are outside the batch.")
    sites = np.empty((motif_width, n_hits), dtype=np.uint8)
    for h in range(n_hits):
        seq = batch[coll.seq_indices[h]]
        start = coll.starts[h] + site_offset
        if start < 0 or start > len(seq) or motif_width > len(seq) - start:
            raise InvariantError(
                f"site {h}: window [start={start}, width={motif_width}] exceeds sequence length {len(seq)}."
            )
        sites[:, h] = seq[start : start + motif_width]
        if coll.strands[h] == 1:
            sites[:, h] = reverse_complement(sites[:, h])
    return sites


def site_strings(sites):
    return ["".join("ACGTN"[min(c, 4)] for c in sites[:, h]) for h in range(sites.shape[1])]


def build_pcm(sites, motif_width):
    if motif_width <= 0:
        raise ValueError("motif_width must be positive.")
    if sites.shape[0] != motif_width:
        raise ValueError("sites row count must equal motif_width.")
    if np.any(sites > N_CODE):
        raise ValueError("sites contain invalid DNA codes.")
    pcm = np.zeros((4, motif_width), dtype=np.float32)
    for h in range(sites.shape[1]):
        for p in range(motif_width):
            code = sites[p, h]
            if code < N_CODE:
                pcm[code, p] += 1.0
    return pcm


def pcm_to_pfm(pcm, pseudocount=0.25):
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.shape[0] != NUCLEOTIDE_CARDINALITY:
        raise ModelDimensionError(f"PCM must have 4 rows, got {pcm.shape[0]}.")
    if not np.all(np.isfinite(pcm)):
        raise ModelFormatError("", "PCM contains non-finite values.")
    if np.any(pcm < 0):
        raise ModelFormatError("", "PCM values must be non-negative.")
    if not (np.isfinite(pseudocount) and pseudocount >= 0):
        raise ModelFormatError("", "pseudocount must be finite and non-negative.")
    pc = np.float32(pseudocount)
    n_sites = pcm.sum(axis=0, dtype=np.float64)
    if pc == 0 and np.any(n_sites == 0):
        raise ModelFormatError(
            "", "PCM columns with zero observations require a positive pseudocount."
        )
    denom = n_sites + NUCLEOTIDE_CARDINALITY * pc
    return ((pcm + pc) / denom).astype(np.float32)


def reconstruct_pfm(model, sequences, selector, *, pseudocount=0.25, strands="best"):
    coll = select_sites(model, sequences, selector, strands=strands)
    if len(coll) == 0:
        raise ValueError("No sites found for PFM reconstruction.")
    motif_width = model.motif_length
    offset = site_start_offset(model)
    sites = extract_site_matrix(sequences, coll, motif_width, site_offset=offset)
    pcm = build_pcm(sites, motif_width)
    return pcm_to_pfm(pcm, pseudocount=pseudocount)
