"""Public scan API: validate input, allocate tracks, dispatch to kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._kernels import (
    batch_pwm_models_forward,
    batch_pwm_models_reverse,
    batch_rolling_forward,
    batch_rolling_forward_parallel,
    batch_rolling_models_forward,
    batch_rolling_models_reverse,
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


@dataclass(slots=True)
class _PackedScanBatch:
    """Packed raw profiles in original model order."""

    forward_data: np.ndarray
    forward_offsets: np.ndarray
    reverse_data: np.ndarray
    reverse_offsets: np.ndarray
    motif_lengths: np.ndarray

    def pair(self, model_index):
        forward_offsets = self.forward_offsets[model_index]
        reverse_offsets = self.reverse_offsets[model_index]
        forward_start, forward_stop = forward_offsets[0], forward_offsets[-1]
        reverse_start, reverse_stop = reverse_offsets[0], reverse_offsets[-1]
        return StrandPair(
            RaggedArray(
                self.forward_data[forward_start:forward_stop],
                forward_offsets - forward_start,
            ),
            RaggedArray(
                self.reverse_data[reverse_start:reverse_stop],
                reverse_offsets - reverse_start,
            ),
        )


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


def _scan_builtin_model_into(
    model,
    batch,
    forward_data,
    forward_offsets,
    reverse_data,
    reverse_offsets,
    model_index,
    parallel,
):
    """Scan one built-in model into its row of the packed output."""
    if isinstance(model, PWM):
        forward_kernel = batch_scan_forward_parallel if parallel else batch_scan_forward
        reverse_kernel = batch_scan_reverse_parallel if parallel else batch_scan_reverse
        kernel_args = (model.motif_length,)
    elif isinstance(model, (BaMM, Dimont, Slim)):
        forward_kernel = batch_rolling_forward_parallel if parallel else batch_rolling_forward
        reverse_kernel = batch_rolling_reverse_parallel if parallel else batch_rolling_reverse
        kernel_args = (model.order + 1, model.motif_length)
    elif isinstance(model, SiteGA):
        forward_kernel = batch_rolling_forward_parallel if parallel else batch_rolling_forward
        reverse_kernel = batch_rolling_reverse_parallel if parallel else batch_rolling_reverse
        kernel_args = (2, model.motif_length - 1)
    else:
        raise TypeError(f"unsupported built-in scan model: {type(model).__name__}")

    forward_kernel(
        model.weights,
        batch.data,
        batch.offsets,
        forward_data,
        forward_offsets[model_index],
        *kernel_args,
    )
    reverse_kernel(
        model.weights,
        batch.data,
        batch.offsets,
        reverse_data,
        reverse_offsets[model_index],
        *kernel_args,
    )


def _scan_models_batch(models, batch):
    """Scan built-in model groups into one packed forward/reverse batch."""
    if not isinstance(batch, EncodedSequences):
        raise TypeError("batch must be an EncodedSequences batch.")
    models = list(models)
    n_models = len(models)
    n_rows = len(batch)
    if n_models == 0:
        empty_offsets = np.empty((0, n_rows + 1), dtype=np.int64)
        empty_data = np.empty(0, dtype=np.float32)
        return _PackedScanBatch(
            empty_data,
            empty_offsets,
            empty_data.copy(),
            empty_offsets.copy(),
            np.empty(0, dtype=np.int64),
        )

    for model in models:
        _validate_model_contract(model, capability="batch_scan")

    forward_offsets = np.empty((n_models, n_rows + 1), dtype=np.int64)
    reverse_offsets = np.empty_like(forward_offsets)
    cursor = 0
    for model_index, model in enumerate(models):
        local_offsets = _scan_offsets(batch, model)
        forward_offsets[model_index] = local_offsets + cursor
        reverse_offsets[model_index] = local_offsets + cursor
        cursor += int(local_offsets[-1])
    forward_data = np.empty(cursor, dtype=np.float32)
    reverse_data = np.empty(cursor, dtype=np.float32)
    motif_lengths = np.asarray([model.motif_length for model in models], dtype=np.int64)

    groups = {}
    custom = []
    for model_index, model in enumerate(models):
        if isinstance(model, PWM):
            key = ("pwm", model.motif_length, 0)
        elif isinstance(model, (BaMM, Dimont, Slim)):
            key = ("rolling", model.order + 1, model.motif_length)
        elif isinstance(model, SiteGA):
            key = ("sitega", 2, model.motif_length - 1)
        else:
            custom.append(model_index)
            continue
        groups.setdefault(key, []).append(model_index)

    for (kind, first, second), model_indices in groups.items():
        indices = np.asarray(model_indices, dtype=np.int64)
        model_items = int(
            forward_offsets[model_indices[0], -1]
            - forward_offsets[model_indices[0], 0]
        )
        model_parallel = use_parallel(
            model_items,
            rows=n_rows,
            groups=len(model_indices),
        )
        if model_parallel:
            if kind == "pwm":
                max_length = first
                weights = np.stack([models[index].weights for index in model_indices])
                lengths = np.full(len(model_indices), max_length, dtype=np.int64)
                batch_pwm_models_forward(
                    weights,
                    lengths,
                    indices,
                    batch.data,
                    batch.offsets,
                    forward_data,
                    forward_offsets,
                )
                batch_pwm_models_reverse(
                    weights,
                    lengths,
                    indices,
                    batch.data,
                    batch.offsets,
                    reverse_data,
                    reverse_offsets,
                )
            else:
                weights = np.stack([models[index].weights for index in model_indices])
                batch_rolling_models_forward(
                    weights,
                    indices,
                    batch.data,
                    batch.offsets,
                    forward_data,
                    forward_offsets,
                    first,
                    second,
                )
                batch_rolling_models_reverse(
                    weights,
                    indices,
                    batch.data,
                    batch.offsets,
                    reverse_data,
                    reverse_offsets,
                    first,
                    second,
                )
        else:
            row_parallel = use_parallel(model_items, rows=n_rows)
            for model_index in model_indices:
                _scan_builtin_model_into(
                    models[model_index],
                    batch,
                    forward_data,
                    forward_offsets,
                    reverse_data,
                    reverse_offsets,
                    model_index,
                    row_parallel,
                )

    for model_index in custom:
        model = models[model_index]
        for row in range(n_rows):
            start, stop = forward_offsets[model_index, row], forward_offsets[model_index, row + 1]
            if stop > start:
                model.scan_into(batch[row], forward_data[start:stop], reverse_data[start:stop])

    return _PackedScanBatch(
        forward_data,
        forward_offsets,
        reverse_data,
        reverse_offsets,
        motif_lengths,
    )


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
    return _scan_batch_into(model, sequences, strands)
