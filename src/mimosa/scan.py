"""Public scan API: validate input, allocate tracks, dispatch to kernels."""

from __future__ import annotations

import numpy as np

from .arrays import EncodedSequences, RaggedArray, StrandPair
from .errors import InvariantError, ModelFormatError, ModelInterfaceError
from .models import (
    BaMM,
    Dimont,
    PWM,
    Slim,
    _validate_model_contract,
    n_positions,
    site_start_offset,
    window_size,
)
from .randomization import WindowModel, _scan_window_batch

STRAND_POLICIES = ("forward", "reverse", "best", "both")


def _validate_strand_policy(strands):
    if strands not in STRAND_POLICIES:
        raise ValueError(
            f"unsupported strand policy: {strands!r}; expected one of {STRAND_POLICIES}."
        )


def _scan_offsets(batch, model):
    offsets = np.empty(len(batch) + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(len(batch)):
        offsets[i + 1] = offsets[i] + n_positions(model, len(batch[i]))
    return offsets


def _validate_finite_builtin_scores(values, model):
    if not np.all(np.isfinite(values)):
        raise ModelFormatError(
            "", f"{type(model).__name__} scan produced non-finite Float32 scores."
        )


def _validate_custom_scan_output(forward, reverse, model):
    if not (np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))):
        raise ModelInterfaceError(
            "scan",
            type(model).__name__,
            "scan_into must fill both output tracks with finite Float32 scores.",
        )


def _circularize(batch, model):
    """Wrap each sequence so scan index ``i`` is the site starting at base ``i``."""
    left = site_start_offset(model)
    right = window_size(model) - left - 1
    rows = [
        row[np.arange(-left, len(row) + right) % len(row)] if len(row) else row
        for row in (batch[i] for i in range(len(batch)))
    ]
    return EncodedSequences.from_rows(rows)


def _scan_batch_into(model, batch, strands, circular=False):
    _validate_model_contract(model)
    if isinstance(model, (PWM, BaMM, Dimont, Slim)):
        from ._jax_backend import scan_circular, scan_linear

        if circular:
            tracks = scan_circular(model, batch)
            offsets = batch.offsets
        else:
            tracks, offsets = scan_linear(model, batch)
        # Prepared-profile normalization writes into these host buffers.
        fwd, rev = (np.array(track, dtype=np.float32, copy=True) for track in tracks)
        _validate_finite_builtin_scores(fwd, model)
        _validate_finite_builtin_scores(rev, model)
        offsets.setflags(write=False)
        if strands == "forward":
            return RaggedArray(fwd, offsets)
        if strands == "reverse":
            return RaggedArray(rev, offsets)
        if strands == "best":
            return RaggedArray(np.maximum(fwd, rev), offsets)
        return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, offsets))

    if circular:
        sequence_offsets = batch.offsets
        batch = _circularize(batch, model)
    offsets = _scan_offsets(batch, model)
    if circular and not np.array_equal(offsets, sequence_offsets):
        raise InvariantError("circular scan must yield one score per sequence base.")
    if isinstance(model, WindowModel):
        return _scan_window_batch(model, batch, offsets, strands)
    fwd = np.full(int(offsets[-1]), np.nan, dtype=np.float32)
    rev = np.full(int(offsets[-1]), np.nan, dtype=np.float32)
    for row in range(len(batch)):
        start, stop = offsets[row], offsets[row + 1]
        if stop > start:
            model.scan_into(batch[row], fwd[start:stop], rev[start:stop])
    _validate_custom_scan_output(fwd, rev, model)
    offsets.setflags(write=False)
    if strands == "forward":
        return RaggedArray(fwd, offsets)
    if strands == "reverse":
        return RaggedArray(rev, offsets)
    if strands == "best":
        return RaggedArray(np.maximum(fwd, rev), offsets)
    return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, offsets))


def scan(model, sequences, *, strands="forward"):
    """Scan a motif model against an EncodedSequences batch.

    Returns a RaggedArray for forward/reverse/best, or a StrandPair for both.
    Built-in models use JAX; custom models use their own ``scan_into`` method.
    """
    _validate_strand_policy(strands)
    if not isinstance(sequences, EncodedSequences):
        raise TypeError("sequences must be an EncodedSequences batch.")
    return _scan_batch_into(model, sequences, strands)
