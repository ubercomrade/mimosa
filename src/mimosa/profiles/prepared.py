"""ScoreProfile and PreparedProfile."""

from __future__ import annotations

import numpy as np

from ..arrays import RaggedArray, StrandPair
from ..errors import ModelFormatError
from ..models import MotifModel, site_start_offset
from .anchors import AnchorCSR, collect_both_anchors
from .normalization import (
    HybridEmpiricalLogTail,
    _fit_exact,
    normalize_bundle,
    normalization_fingerprint,
)


def _as_float32_min_logerr(value):
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            threshold = np.float32(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_logerr must be a finite Float32 value.") from exc
    if not np.isfinite(threshold):
        raise ValueError("min_logerr must be a finite Float32 value.")
    return threshold


def _freeze_prepared_arrays(bundle, anchors):
    strands = (bundle.forward, bundle.reverse)
    for strand in strands:
        strand.data.setflags(write=False)
        strand.offsets.setflags(write=False)
    for anchor_set in anchors:
        anchor_set.positions.setflags(write=False)
        anchor_set.offsets.setflags(write=False)


def _all_finite(values, chunk_size=1 << 20):
    return all(
        np.all(np.isfinite(values[start : start + chunk_size]))
        for start in range(0, values.size, chunk_size)
    )


class ScoreProfile:
    """Precomputed per-position score profiles; both strands share the scores."""

    __slots__ = ("name", "scores")

    def __init__(self, name, scores):
        if not name:
            raise ValueError("ScoreProfile name must not be empty.")
        if not isinstance(scores, RaggedArray):
            if isinstance(scores, np.ndarray) and scores.ndim != 2:
                raise TypeError(
                    "scores must be a RaggedArray or a two-dimensional score array."
                )
            try:
                scores = RaggedArray.from_rows(scores)
            except (TypeError, ValueError, IndexError) as exc:
                raise TypeError(
                    "scores must be a RaggedArray or a sequence of score rows."
                ) from exc
        if not _all_finite(scores.data):
            raise ModelFormatError("", "score profile contains non-finite values.")
        self.name = str(name)
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __eq__(self, other):
        return (
            isinstance(other, ScoreProfile)
            and self.name == other.name
            and self.scores == other.scores
        )

    def __repr__(self):
        return f"ScoreProfile({self.name!r}, {len(self.scores)} rows)"


class PreparedProfile:
    """Normalized profile bundle with pre-collected anchors."""

    __slots__ = (
        "name",
        "bundle",
        "anchors",
        "min_logerr",
        "normalization",
        "site_start_offset",
    )

    def __init__(
        self,
        name,
        bundle,
        anchors,
        min_logerr=0.0,
        normalization=None,
        site_start_offset=0,
    ):
        threshold = _as_float32_min_logerr(min_logerr)
        if not isinstance(bundle, StrandPair):
            raise TypeError("prepared profile bundle must be a StrandPair.")
        if (
            not isinstance(anchors, tuple)
            or len(anchors) != 2
            or not all(isinstance(anchor_set, AnchorCSR) for anchor_set in anchors)
        ):
            raise TypeError("prepared profile anchors must be a pair of AnchorCSR values.")
        if (
            isinstance(site_start_offset, bool)
            or not isinstance(site_start_offset, (int, np.integer))
            or site_start_offset < 0
        ):
            raise ValueError("site_start_offset must be a non-negative integer.")
        n_rows = len(bundle.forward)
        if len(bundle.reverse) != n_rows:
            raise ValueError("prepared strand bundles must have equal row counts.")
        if not np.array_equal(bundle.forward.offsets, bundle.reverse.offsets):
            raise ValueError("prepared strand bundles must have identical row layouts.")
        for strand in (bundle.forward, bundle.reverse):
            if strand.data.dtype != np.dtype(np.float32):
                raise TypeError("prepared scores must use Float32 storage.")
            if not _all_finite(strand.data):
                raise ValueError("prepared scores must be finite.")
        if anchors[0].offsets.size != n_rows + 1:
            raise ValueError("forward anchor rows do not match the profile bundle.")
        if anchors[1].offsets.size != n_rows + 1:
            raise ValueError("reverse anchor rows do not match the profile bundle.")
        for csr, strand in zip(anchors, (bundle.forward, bundle.reverse)):
            for row in range(n_rows):
                start, stop = csr.offsets[row], csr.offsets[row + 1]
                positions = csr.positions[start:stop]
                if np.any(
                    (positions < site_start_offset)
                    | (positions >= site_start_offset + len(strand[row]))
                ):
                    raise ValueError("anchor position is outside its profile row.")
        self.name = str(name)
        self.bundle = bundle
        self.anchors = anchors
        self.min_logerr = threshold
        self.normalization = normalization if normalization is not None else HybridEmpiricalLogTail()
        try:
            normalization_fingerprint(self.normalization)
        except ValueError as exc:
            raise ValueError("prepared profile normalization is unsupported.") from exc
        self.site_start_offset = int(site_start_offset)
        _freeze_prepared_arrays(bundle, anchors)

    def __eq__(self, other):
        return (
            isinstance(other, PreparedProfile)
            and self.name == other.name
            and self.min_logerr == other.min_logerr
            and self.normalization == other.normalization
            and self.site_start_offset == other.site_start_offset
            and self.bundle == other.bundle
            and self.anchors[0] == other.anchors[0]
            and self.anchors[1] == other.anchors[1]
        )


def prepare_profile(model_or_scores, sequences=None, *, background=None, min_logerr=0.0, normalization=None, cache=None):
    return _prepare_profile(
        model_or_scores,
        sequences,
        background=background,
        min_logerr=min_logerr,
        normalization=normalization,
        cache=cache,
    )


def _prepare_profile(
    model_or_scores,
    sequences=None,
    *,
    background=None,
    min_logerr=0.0,
    normalization=None,
    cache=None,
    _preparation_context=None,
):
    """Prepare a profile for repeated comparison.

    Accepts a ScoreProfile (no sequences) or a MotifModel plus EncodedSequences.
    """
    threshold = _as_float32_min_logerr(min_logerr)
    if normalization is None:
        normalization = HybridEmpiricalLogTail()

    if cache is not None:
        from ..cache import _cached_prepared_profile

        key, cached = _cached_prepared_profile(
            cache,
            model_or_scores,
            sequences,
            background,
            threshold,
            normalization,
            _preparation_context,
        )
        if cached is not None:
            return cached
    else:
        key = None

    calibration = None
    if isinstance(model_or_scores, ScoreProfile):
        if sequences is not None:
            raise ValueError("ScoreProfile preparation does not consume sequences.")
        raw = StrandPair(model_or_scores.scores, model_or_scores.scores)
        name = model_or_scores.name
        profile_site_start_offset = 0
    elif isinstance(model_or_scores, MotifModel):
        if sequences is None:
            raise ValueError("motif prepared profiles require comparison sequences.")
        from ..scan import _scan_batch_into

        raw = _scan_batch_into(model_or_scores, sequences, "both")
        name = model_or_scores.name
        profile_site_start_offset = site_start_offset(model_or_scores)
        bg = sequences if background is None else background
        if bg is not sequences:
            bg_raw = _scan_batch_into(model_or_scores, bg, "both")
            calibration = bg_raw
            del bg_raw
    else:
        raise TypeError(
            "model_or_scores must be a ScoreProfile or a MotifModel."
        )

    table = _fit_exact(normalization, raw if calibration is None else calibration)
    del calibration
    norm_bundle = normalize_bundle(
        table,
        raw,
        in_place=isinstance(model_or_scores, MotifModel),
    )
    del raw, table
    if cache is not None:
        from ..cache import _store_normalized_profile

        _store_normalized_profile(
            cache,
            key,
            name,
            norm_bundle,
            normalization,
            profile_site_start_offset,
        )
    anchors = collect_both_anchors(
        norm_bundle, threshold, position_offset=profile_site_start_offset
    )
    prepared = PreparedProfile(
        name,
        norm_bundle,
        anchors,
        threshold,
        normalization,
        profile_site_start_offset,
    )
    return prepared
