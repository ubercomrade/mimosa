"""ScoreProfile and PreparedProfile."""

from __future__ import annotations

import numpy as np

from ..arrays import RaggedArray, StrandPair
from ..errors import ModelFormatError
from ..models import MotifModel
from .anchors import collect_both_anchors
from .normalization import (
    HybridEmpiricalLogTail,
    _fit_normalize,
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
        if not np.all(np.isfinite(scores.data)):
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

    __slots__ = ("name", "bundle", "anchors", "min_logerr", "normalization")

    @classmethod
    def _from_validated(cls, name, bundle, anchors, min_logerr=0.0, normalization=None):
        """Build a profile after the internal normalization/anchor pipeline."""
        profile = cls.__new__(cls)
        profile.name = str(name)
        profile.bundle = bundle
        profile.anchors = anchors
        profile.min_logerr = np.float32(min_logerr)
        profile.normalization = (
            normalization if normalization is not None else HybridEmpiricalLogTail()
        )
        return profile

    def __init__(self, name, bundle, anchors, min_logerr=0.0, normalization=None):
        if not np.isfinite(min_logerr):
            raise ValueError("min_logerr must be finite.")
        n_rows = len(bundle.forward)
        if len(bundle.reverse) != n_rows:
            raise ValueError("prepared strand bundles must have equal row counts.")
        if anchors[0].offsets.size != n_rows + 1:
            raise ValueError("forward anchor rows do not match the profile bundle.")
        if anchors[1].offsets.size != n_rows + 1:
            raise ValueError("reverse anchor rows do not match the profile bundle.")
        for csr, strand in zip(anchors, (bundle.forward, bundle.reverse)):
            for row in range(n_rows):
                start, stop = csr.offsets[row], csr.offsets[row + 1]
                positions = csr.positions[start:stop]
                if np.any((positions < 0) | (positions >= len(strand[row]))):
                    raise ValueError("anchor position is outside its profile row.")
        self.name = str(name)
        self.bundle = bundle
        self.anchors = anchors
        self.min_logerr = np.float32(min_logerr)
        self.normalization = normalization if normalization is not None else HybridEmpiricalLogTail()

    def __eq__(self, other):
        return (
            isinstance(other, PreparedProfile)
            and self.name == other.name
            and self.min_logerr == other.min_logerr
            and self.normalization == other.normalization
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
    threshold = np.float32(min_logerr)
    if not np.isfinite(threshold):
        raise ValueError("min_logerr must be finite.")
    if normalization is None:
        normalization = HybridEmpiricalLogTail()

    if cache is not None:
        from ..cache import _cached_prepared_profile, _store_prepared_profile

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

    if isinstance(model_or_scores, ScoreProfile):
        if sequences is not None:
            raise ValueError("ScoreProfile preparation does not consume sequences.")
        raw = StrandPair(model_or_scores.scores, model_or_scores.scores)
        _, norm_bundle = _fit_normalize(
            normalization, raw, tail_logerr=threshold
        )
        anchors = collect_both_anchors(norm_bundle, threshold)
        prepared = PreparedProfile._from_validated(
            model_or_scores.name, norm_bundle, anchors, threshold, normalization
        )
        if cache is not None:
            prepared = _store_prepared_profile(cache, key, prepared)
        return prepared

    if isinstance(model_or_scores, MotifModel):
        if sequences is None:
            raise ValueError("motif prepared profiles require comparison sequences.")
        from ..scan import _scan_batch_into

        raw = _scan_batch_into(model_or_scores, sequences, "both")
        bg = sequences if background is None else background
        if bg is sequences:
            _, norm_bundle = _fit_normalize(
                normalization, raw, tail_logerr=threshold
            )
        else:
            bg_raw = _scan_batch_into(model_or_scores, bg, "both")
            _, norm_bundle = _fit_normalize(
                normalization,
                raw,
                calibration=bg_raw,
                tail_logerr=threshold,
            )
        anchors = collect_both_anchors(norm_bundle, threshold)
        prepared = PreparedProfile._from_validated(
            model_or_scores.name, norm_bundle, anchors, threshold, normalization
        )
        if cache is not None:
            prepared = _store_prepared_profile(cache, key, prepared)
        return prepared

    raise TypeError(
        "model_or_scores must be a ScoreProfile or a MotifModel."
    )
