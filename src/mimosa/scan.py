"""Public scan API: validate input, allocate tracks, dispatch to kernels."""

from __future__ import annotations

import numpy as np

from ._kernels import (
    batch_rolling_forward,
    batch_rolling_forward_parallel,
    batch_rolling_reverse,
    batch_rolling_reverse_parallel,
    batch_scan_forward,
    batch_scan_forward_parallel,
    batch_scan_reverse,
    batch_scan_reverse_parallel,
    best_strand_reduce,
)
from .arrays import EncodedSequences, RaggedArray, StrandPair
from .errors import ModelInterfaceError
from .models import (
    BaMM,
    Dimont,
    MotifModel,
    PWM,
    SiteGA,
    Slim,
    _validate_model_contract,
    n_positions,
)
from .parallel import use_parallel

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


def _scan_batch_into(model, batch, strands):
    _validate_model_contract(model)
    offsets = _scan_offsets(batch, model)
    data = np.empty(int(offsets[-1]), dtype=np.float32)
    parallel = use_parallel(int(offsets[-1]), rows=len(batch))
    sfwd = batch_scan_forward_parallel if parallel else batch_scan_forward
    srev = batch_scan_reverse_parallel if parallel else batch_scan_reverse
    rfwd = batch_rolling_forward_parallel if parallel else batch_rolling_forward
    rrev = batch_rolling_reverse_parallel if parallel else batch_rolling_reverse

    if isinstance(model, PWM):
        forward_kernel, reverse_kernel = sfwd, srev
        kernel_args = (model.motif_length,)
    elif isinstance(model, (BaMM, Dimont, Slim)):
        forward_kernel, reverse_kernel = rfwd, rrev
        kernel_args = (model.order + 1, model.motif_length)
    elif isinstance(model, SiteGA):
        forward_kernel, reverse_kernel = rfwd, rrev
        kernel_args = (2, model.motif_length - 1)
    else:
        fwd = np.empty_like(data)
        rev = np.empty_like(data)
        for row in range(len(batch)):
            start, stop = offsets[row], offsets[row + 1]
            if stop > start:
                model.scan_into(batch[row], fwd[start:stop], rev[start:stop])
        if strands == "forward":
            data[:] = fwd
        elif strands == "reverse":
            data[:] = rev
        elif strands == "best":
            best_strand_reduce(fwd, rev, data)
        else:
            return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, offsets.copy()))
        return RaggedArray(data, offsets)

    if strands == "forward":
        forward_kernel(model.weights, batch.data, batch.offsets, data, offsets, *kernel_args)
    elif strands == "reverse":
        reverse_kernel(model.weights, batch.data, batch.offsets, data, offsets, *kernel_args)
    else:
        fwd = np.empty_like(data)
        rev = np.empty_like(data)
        forward_kernel(model.weights, batch.data, batch.offsets, fwd, offsets, *kernel_args)
        reverse_kernel(model.weights, batch.data, batch.offsets, rev, offsets, *kernel_args)
        if strands == "best":
            best_strand_reduce(fwd, rev, data)
        else:
            return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, offsets.copy()))
    return RaggedArray(data, offsets)


def scan(model, sequences, *, strands="forward"):
    """Scan a motif model against an EncodedSequences batch.

    Returns a RaggedArray for forward/reverse/best, or a StrandPair for both.
    Large batches automatically use Numba parallel kernels; small batches stay
    on the lower-overhead serial path.
    """
    _validate_strand_policy(strands)
    if not isinstance(model, MotifModel):
        raise ModelInterfaceError(
            "compare", type(model).__name__, "model must be a MotifModel."
        )
    if not isinstance(sequences, EncodedSequences):
        raise TypeError("sequences must be an EncodedSequences batch.")
    if strands == "both":
        return _scan_batch_into(model, sequences, "both")
    return _scan_batch_into(model, sequences, strands)


def scan_result_lengths(model, sequences):
    return np.array(
        [n_positions(model, len(sequences[i])) for i in range(len(sequences))],
        dtype=np.int64,
    )
