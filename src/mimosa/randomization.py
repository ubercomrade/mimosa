"""Fixed-window scorers and coherent, position-specific alphabet relabeling.

An input-relabel surrogate evaluates f(pi(x)), using one fixed table per motif.
The single-window score distribution is invariant on uniform iid DNA; neither
overlapping-window dependence nor a biological null hypothesis is guaranteed.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .arrays import EncodedSequences, RaggedArray, StrandPair
from .errors import ModelInterfaceError
from .models import (
    PWM, BaMM, Dimont, MotifModel, SiteGA, Slim,
    _require_sitega_enabled, _validate_model_contract, strict_integer, window_size,
)

INPUT_RELABEL_VERSION = "input-alphabet-relabel-v1"
INPUT_RELABEL_STRATEGY = "profile-input-relabel"
DEFAULT_WINDOW_BATCH_SIZE = 8192
_MAX_WINDOW_ELEMENTS = 1_048_576


def _reverse_complement_windows(windows):
    result = np.array(windows[:, ::-1], dtype=np.uint8, order="C", copy=True)
    known = result < 4
    result[known] = 3 - result[known]
    return result


class WindowModel(MotifModel):
    """Adapt a deterministic batched function of oriented DNA windows.

    ``score_windows`` receives an (n, width) uint8 array, with A/C/G/T/N encoded
    0/1/2/3/4, and returns n finite real scores. ``width`` includes both context
    flanks. Scores must not depend on batch size or other windows in the batch.
    A GPU callback can convert the input and return a CPU NumPy score vector.

    ``fingerprint`` identifies the function and learned parameters for caching
    and null construction. ``calibration_group`` identifies a common scoring
    formalism/configuration for null construction, not the identity of one TF.
    Changing either requires a different identifier. Neither is inferred from
    a Python callable's name or bytecode.
    """

    def __init__(
        self, name, motif_length, score_windows, *, left_context=0,
        right_context=0, fingerprint=None, calibration_group=None,
        batch_size=DEFAULT_WINDOW_BATCH_SIZE,
    ):
        self.name = name
        self.motif_length = strict_integer(motif_length, "motif_length")
        self.left_context = strict_integer(left_context, "left_context")
        self.right_context = strict_integer(right_context, "right_context")
        self.batch_size = strict_integer(batch_size, "batch_size")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not callable(score_windows):
            raise TypeError("score_windows must be callable.")
        for field, value in (("fingerprint", fingerprint), ("calibration_group", calibration_group)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be a non-empty string when supplied.")
        self._score_windows = score_windows
        self._fingerprint = fingerprint
        self.calibration_group = calibration_group
        _validate_model_contract(self)

    def _evaluate_windows(self, windows):
        raw = np.asarray(self._score_windows(windows))
        if raw.shape != (len(windows),) or raw.dtype.kind not in "biuf":
            raise ModelInterfaceError(
                "scan", type(self).__name__,
                "score_windows must return one real score per input window.",
            )
        with np.errstate(over="ignore", invalid="ignore"):
            result = raw.astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise ModelInterfaceError(
                "scan", type(self).__name__,
                "score_windows must return finite Float32 scores.",
            )
        return result

    def scan_into(self, sequence, forward, reverse, /):
        width = window_size(self)
        windows = np.lib.stride_tricks.sliding_window_view(sequence, width)
        batch_size = min(self.batch_size, max(1, _MAX_WINDOW_ELEMENTS // width))
        for start in range(0, len(windows), batch_size):
            stop = start + batch_size
            block = windows[start:stop]
            forward[start:stop] = self._evaluate_windows(block)
            reverse[start:stop] = self._evaluate_windows(_reverse_complement_windows(block))

    def fingerprint(self):
        if self._fingerprint is None:
            return None
        identity = (
            f"window-model-v1|{self.name}|{self.motif_length}|{self.left_context}|"
            f"{self.right_context}|{self.calibration_group}|{self._fingerprint}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()


def _scan_window_batch(model, batch, offsets, strands):
    """Batch windows across rows, retaining the scanner's exact row geometry."""
    width = window_size(model)
    batch_size = min(model.batch_size, max(1, _MAX_WINDOW_ELEMENTS // width))
    count = int(offsets[-1])
    forward = np.empty(count, np.float32) if strands != "reverse" else None
    reverse = np.empty(count, np.float32) if strands != "forward" else None
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        positions = np.arange(start, stop, dtype=np.int64)
        rows = np.searchsorted(offsets, positions, side="right") - 1
        starts = batch.offsets[rows] + positions - offsets[rows]
        windows = batch.data[starts[:, None] + np.arange(width)[None, :]]
        if forward is not None:
            forward[start:stop] = model._evaluate_windows(windows)
        if reverse is not None:
            reverse[start:stop] = model._evaluate_windows(_reverse_complement_windows(windows))
    if strands == "forward":
        return RaggedArray(forward, offsets)
    if strands == "reverse":
        return RaggedArray(reverse, offsets)
    if strands == "best":
        np.maximum(forward, reverse, out=forward)
        return RaggedArray(forward, offsets)
    return StrandPair(RaggedArray(forward, offsets), RaggedArray(reverse, offsets))


def alphabet_permutations(motif_length, seed=127):
    """Draw one reproducible A/C/G/T permutation per motif position.

    The unused column permutation preserves the seed stream of the historical
    PWM shuffle and the matched BaMM validation experiment. Positions are not
    reordered by input relabeling.
    """
    length = strict_integer(motif_length, "motif_length")
    seed = strict_integer(seed, "seed")
    if length < 1 or seed < 0:
        raise ValueError("motif_length must be positive and seed non-negative.")
    rng = np.random.default_rng(seed)
    rng.permutation(length)
    result = np.array([rng.permutation(4) for _ in range(length)], dtype=np.uint8)
    result.setflags(write=False)
    return result


def _validate_permutations(model, permutations):
    raw = np.asarray(permutations)
    expected = (model.motif_length, 4)
    if raw.shape != expected or raw.dtype.kind not in "iu":
        raise ValueError(f"permutations must be an integer array with shape {expected}.")
    if not np.array_equal(np.sort(raw, axis=1), np.tile(np.arange(4), (len(raw), 1))):
        raise ValueError("each permutation row must contain A/C/G/T codes 0,1,2,3 exactly once.")
    result = np.array(raw, dtype=np.uint8, order="C", copy=True)
    result.setflags(write=False)
    return result


class InputRelabeledModel(WindowModel):
    """Generic f(pi(x)) wrapper; reverse-complement precedes input recoding.

    Only the motif coordinates are recoded; declared context flanks and N stay
    fixed. The source model and its learned parameters are unchanged. Built-in
    models are faster through ``relabel_model``, which compiles the same input
    transformation into an equivalent native table permutation.
    """

    def __init__(self, model, permutations, *, batch_size=DEFAULT_WINDOW_BATCH_SIZE):
        _validate_model_contract(model)
        self.source_model = model
        self._permutations = _validate_permutations(model, permutations)
        full = np.tile(np.arange(5, dtype=np.uint8), (window_size(model), 1))
        full[model.left_context:model.left_context + model.motif_length, :4] = self.permutations
        full.setflags(write=False)
        self._full_permutations = full
        super().__init__(
            model.name, model.motif_length, self._recoded_scores,
            left_context=model.left_context, right_context=model.right_context,
            batch_size=batch_size,
        )

    @property
    def permutations(self):
        return self._permutations

    def _recoded_scores(self, windows):
        transformed = self._full_permutations[np.arange(windows.shape[1])[None, :], windows]
        if isinstance(self.source_model, WindowModel):
            return self.source_model._evaluate_windows(transformed)
        from .scan import scan

        width = transformed.shape[1]
        encoded = EncodedSequences(
            transformed.reshape(-1), np.arange(len(transformed) + 1, dtype=np.int64) * width,
        )
        return scan(self.source_model, encoded, strands="forward").data

    def fingerprint(self):
        from .io.bundles import model_fingerprint

        original = model_fingerprint(self.source_model)
        digest = hashlib.sha256(f"{INPUT_RELABEL_VERSION}|{original}|".encode())
        digest.update(self.permutations.tobytes())
        return digest.hexdigest()


def _native_relabel_weights(model, permutations, kmer_size):
    """Permute state labels coherently at every occurrence in a k-mer term."""
    codes = np.arange(model.weights.shape[0], dtype=np.int64)
    result = np.empty_like(model.weights)
    for term in range(model.weights.shape[1]):
        mapped = np.zeros_like(codes)
        for offset in range(kmer_size):
            power = 5 ** (kmer_size - offset - 1)
            digits = (codes // power) % 5
            position = term - model.left_context + offset
            if 0 <= position < model.motif_length:
                mapping = np.r_[permutations[position], np.uint8(4)].astype(np.int64)
                digits = mapping[digits]
            mapped += digits * power
        result[:, term] = model.weights[mapped, term]
    return result


def relabel_model(model, *, seed=127, permutations=None, batch_size=DEFAULT_WINDOW_BATCH_SIZE):
    """Create an input-relabel surrogate, using native built-in tables if possible.

    Supplying ``permutations`` allows a shared randomization across projections
    or an exact replay. The seed is used only when permutations are omitted.
    The return value is a normal native model or an InputRelabeledModel; both
    implement the same function of an oriented full window.
    """
    _validate_model_contract(model)
    if permutations is None:
        permutations = alphabet_permutations(model.motif_length, seed)
    permutations = _validate_permutations(model, permutations)
    if isinstance(model, PWM):
        weights = _native_relabel_weights(model, permutations, 1)
        return PWM(model.name, weights, model.background)
    if isinstance(model, (BaMM, Dimont, Slim)):
        weights = _native_relabel_weights(model, permutations, model.order + 1)
        return type(model)(model.name, weights, model.order, model.motif_length)
    if isinstance(model, SiteGA):
        weights = _native_relabel_weights(model, permutations, 2)
        return SiteGA(model.name, weights, model.motif_length)
    return InputRelabeledModel(model, permutations, batch_size=batch_size)


def model_calibration_group(model):
    """Return the declared model kind/order, not a guarantee of calibration."""
    if isinstance(model, InputRelabeledModel):
        return model_calibration_group(model.source_model)
    if isinstance(model, PWM):
        return "PWM"
    if isinstance(model, (BaMM, Dimont, Slim)):
        return f"{type(model).__name__}:order={model.order}"
    if isinstance(model, SiteGA):
        _require_sitega_enabled()
        return "SiteGA"
    declared = getattr(model, "calibration_group", None)
    if declared is not None:
        if not isinstance(declared, str) or not declared:
            raise ValueError("calibration_group must be a non-empty string.")
        return f"custom:{declared}"
    raise ValueError(
        "custom models require an explicit calibration_group for null construction/application."
    )
