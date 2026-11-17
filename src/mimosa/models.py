"""Motif model types: PWM, BaMM, SiteGA, Dimont, Slim, and the custom contract."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .arrays import EncodedSequences, RaggedArray, StrandPair
from .errors import (
    InvariantError, MimosaError, ModelDimensionError, ModelFormatError, ModelInterfaceError,
)

NUCLEOTIDE_CARDINALITY = 4
PSEUDOCOUNT_PWM = 1e-4
MAX_CONTEXT_ORDER = 10
MAX_CONTEXT_REPRESENTATION_ELEMENTS = 100_000_000
MAX_DIMONT_SLIM_ORDER = 6
MAX_DIMONT_SLIM_LENGTH = 30
_SITEGA_ENABLED = False  # Keep the implementation for future validation and restoration.

STRAND_POLICIES = ("forward", "reverse", "best", "both")


def _require_sitega_enabled():
    if not _SITEGA_ENABLED:
        raise MimosaError("SiteGA is temporarily disabled pending further development.")


def _validate_model_name(name, model_type):
    if not isinstance(name, str) or not name:
        raise ValueError(f"{model_type} name must be a non-empty string.")


def strict_integer(value, field, *, allow_text=False):
    """Return an integer without silently truncating numeric geometry values."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if allow_text and isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    raise TypeError(f"{field} must be an integer.")


def _validate_pwm_background(background):
    bg = np.asarray(background, dtype=np.float64)
    if bg.shape != (4,):
        raise ModelFormatError("", "PWM background must have 4 values.")
    for i, v in enumerate(bg):
        if not math.isfinite(v):
            raise ModelFormatError("", f"PWM background[{i}] is not finite.")
        if v <= 0:
            raise ModelFormatError("", f"PWM background[{i}] must be positive.")
    bg_sum = bg.sum()
    if not math.isclose(bg_sum, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelFormatError(
            "", f"PWM background sum is {bg_sum}, expected approximately 1.0."
        )


def _validate_pwm_weights(weights, background):
    if weights.ndim != 2:
        raise ModelDimensionError(
            f"PWM weights must be two-dimensional, got {weights.ndim} dimensions."
        )
    if weights.shape[0] != 5:
        raise ModelDimensionError(
            f"PWM weights must have 5 rows (A,C,G,T,N), got {weights.shape[0]}."
        )
    if weights.shape[1] < 1:
        raise ModelDimensionError(
            f"PWM motif length must be positive, got {weights.shape[1]}."
        )
    if not np.all(np.isfinite(weights)):
        raise ModelFormatError("", "PWM weights contain non-finite values.")
    _validate_pwm_background(background)


def _validate_context_model(representation, context, motif_length, model_name, context_name):
    context = strict_integer(context, f"{model_name} {context_name}")
    motif_length = strict_integer(motif_length, f"{model_name} motif_length")
    if representation.ndim != 2:
        raise ModelDimensionError(
            f"{model_name} representation must be two-dimensional, got {representation.ndim} dimensions."
        )
    if context < 0:
        raise ModelDimensionError(
            f"{model_name} {context_name} must be non-negative, got {context}."
        )
    if context > MAX_CONTEXT_ORDER:
        raise ModelDimensionError(
            f"{model_name} {context_name} must be <= {MAX_CONTEXT_ORDER} to avoid allocation blow-up, got {context}."
        )
    if model_name in ("Dimont", "Slim"):
        if context > MAX_DIMONT_SLIM_ORDER:
            raise ModelDimensionError(
                f"{model_name} {context_name} must be <= {MAX_DIMONT_SLIM_ORDER}, got {context}."
            )
        if motif_length > MAX_DIMONT_SLIM_LENGTH:
            raise ModelDimensionError(
                f"{model_name} motif_length must be <= {MAX_DIMONT_SLIM_LENGTH}, got {motif_length}."
            )
    expected_rows = 5 ** (context + 1)
    if expected_rows * motif_length > MAX_CONTEXT_REPRESENTATION_ELEMENTS:
        raise ModelDimensionError(
            f"{model_name} representation is too large: {expected_rows * motif_length} elements, "
            f"limit is {MAX_CONTEXT_REPRESENTATION_ELEMENTS}."
        )
    if representation.shape[0] != expected_rows:
        raise ModelDimensionError(
            f"{model_name} representation must have {expected_rows} rows for {context_name}={context}, got {representation.shape[0]}."
        )
    if representation.shape[1] != motif_length:
        raise ModelDimensionError(
            f"{model_name} representation columns ({representation.shape[1]}) must match motif_length ({motif_length})."
        )
    if motif_length <= 0:
        raise ModelDimensionError(
            f"{model_name} motif_length must be positive, got {motif_length}."
        )
    if not np.all(np.isfinite(representation)):
        raise ModelFormatError("", f"{model_name} representation contains non-finite values.")
    return context, motif_length


def _validate_sitega(representation, motif_length):
    motif_length = strict_integer(motif_length, "SiteGA motif_length")
    if representation.ndim != 2:
        raise ModelDimensionError(
            f"SiteGA representation must be two-dimensional, got {representation.ndim} dimensions."
        )
    if representation.shape[0] != 25:
        raise ModelDimensionError(
            f"SiteGA representation must have 25 rows (5×5 dinucleotides), got {representation.shape[0]}."
        )
    if representation.shape[1] != motif_length:
        raise ModelDimensionError(
            f"SiteGA representation columns ({representation.shape[1]}) must match motif_length ({motif_length})."
        )
    if motif_length < 2:
        raise ModelDimensionError(
            f"SiteGA motif_length must be at least 2, got {motif_length}."
        )
    if not np.all(np.isfinite(representation)):
        raise ModelFormatError("", "SiteGA representation contains non-finite values.")
    return motif_length


def _as_float32_readonly(arr):
    arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    arr.setflags(write=False)
    return arr


def _reverse_complement_windows(windows):
    reversed_windows = windows[:, ::-1]
    return jnp.where(reversed_windows < 4, 3 - reversed_windows, 4)


class MotifModel(ABC):
    """Model geometry and identity shared by built-in and custom models."""

    name: str
    motif_length: int
    left_context: int = 0
    right_context: int = 0
    # Built-in models keep sequence-level JAX kernels as a fast path; generic
    # scorers are window-based and receive explicitly circularized sequences.
    sequence_scanner = False

    def fingerprint(self):
        return None

    @abstractmethod
    def score_sites(self, windows, /):
        """Score forward-oriented windows: one real JAX score per window.

        ``windows`` is a JAX array of shape ``(n, window_size(self))`` with
        A/C/G/T/N encoded 0..4. The function must be traceable by ``jax.jit``
        and return a finite real score for every window; the reverse strand is
        scored by the same method on reverse-complemented windows.
        """

    def scan(self, sequences, *, strands, circular=False):
        """Return ragged score tracks for a validated sequence batch."""
        _validate_model_contract(self)
        if strands not in STRAND_POLICIES:
            raise ValueError(
                f"unsupported strand policy: {strands!r}; expected one of {STRAND_POLICIES}."
            )
        if not isinstance(sequences, EncodedSequences):
            raise TypeError("sequences must be an EncodedSequences batch.")
        sequence_offsets = sequences.offsets
        if circular and not self.sequence_scanner:
            sequences = self._circularize(sequences)
        forward, reverse, offsets = self._scan_tracks(sequences, strands, circular)
        if (
            circular and not self.sequence_scanner
            and not np.array_equal(offsets, sequence_offsets)
        ):
            raise InvariantError("circular scan must yield one score per sequence base.")
        offsets.setflags(write=False)
        needs_forward = strands in ("forward", "best", "both")
        needs_reverse = strands in ("reverse", "best", "both")
        if (needs_forward and forward is None) or (needs_reverse and reverse is None):
            raise InvariantError("scan implementation must provide the requested strand tracks.")
        if needs_forward:
            forward = np.asarray(forward, dtype=np.float32)
            if not np.all(np.isfinite(forward)):
                raise ModelFormatError(
                    "", f"{type(self).__name__} scan produced non-finite Float32 scores."
                )
        if needs_reverse:
            reverse = np.asarray(reverse, dtype=np.float32)
            if not np.all(np.isfinite(reverse)):
                raise ModelFormatError(
                    "", f"{type(self).__name__} scan produced non-finite Float32 scores."
                )
        if strands == "forward":
            return RaggedArray(forward, offsets)
        if strands == "reverse":
            return RaggedArray(reverse, offsets)
        if strands == "best":
            # A materialized JAX track is read-only; build the maximum afresh.
            return RaggedArray(np.maximum(forward, reverse), offsets)
        return StrandPair(RaggedArray(forward, offsets), RaggedArray(reverse, offsets))

    def _circularize(self, sequences):
        """Wrap each sequence so scan index ``i`` is the site starting at base ``i``."""
        left = site_start_offset(self)
        right = window_size(self) - left - 1
        rows = [
            row[np.arange(-left, len(row) + right) % len(row)] if len(row) else row
            for row in (sequences[i] for i in range(len(sequences)))
        ]
        return EncodedSequences.from_rows(rows)

    def _scan_offsets(self, sequences):
        offsets = np.empty(len(sequences) + 1, dtype=np.int64)
        offsets[0] = 0
        for i in range(len(sequences)):
            offsets[i + 1] = offsets[i] + n_positions(self, len(sequences[i]))
        return offsets

    def _scan_tracks(self, sequences, strands, circular):
        """Return ``(forward, reverse, offsets)``; unrequested strands are None."""
        if self.sequence_scanner:
            from .backend import scan_circular, scan_linear

            if circular:
                return *scan_circular(self, sequences), sequences.offsets
            tracks, offsets = scan_linear(self, sequences)
            return tracks[0], tracks[1], offsets

        from .randomization import _MAX_WINDOW_ELEMENTS, DEFAULT_WINDOW_BATCH_SIZE

        width = window_size(self)
        batch_size = min(
            getattr(self, "batch_size", DEFAULT_WINDOW_BATCH_SIZE),
            max(1, _MAX_WINDOW_ELEMENTS // width),
        )
        offsets = self._scan_offsets(sequences)
        count = int(offsets[-1])
        data = jnp.asarray(sequences.data)
        forward = np.empty(count, np.float32) if strands != "reverse" else None
        reverse = np.empty(count, np.float32) if strands != "forward" else None
        score = jax.jit(self.score_sites)

        def evaluate(windows):
            try:
                raw = jnp.asarray(score(windows))
                if raw.shape != (len(windows),) or raw.dtype.kind not in "biuf":
                    raise ValueError("invalid output shape or dtype")
                return np.asarray(raw, dtype=np.float32)
            except Exception as exc:
                raise ModelInterfaceError(
                    "scan", type(self).__name__,
                    "score_sites must be JAX-traceable and return one real score per window.",
                ) from exc

        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            positions = np.arange(start, stop, dtype=np.int64)
            rows = np.searchsorted(offsets, positions, side="right") - 1
            starts = sequences.offsets[rows] + positions - offsets[rows]
            windows = data[jnp.asarray(starts[:, None] + np.arange(width)[None, :])]
            if forward is not None:
                forward[start:stop] = evaluate(windows)
            if reverse is not None:
                reverse[start:stop] = evaluate(_reverse_complement_windows(windows))
        return forward, reverse, offsets


def _validate_model_contract(model, capability="scan"):
    """Validate the public model contract before any geometry is computed."""
    if isinstance(model, SiteGA):
        _require_sitega_enabled()
    if not isinstance(model, MotifModel):
        raise ModelInterfaceError(
            capability, type(model).__name__, "model must be a MotifModel."
        )
    model_type = type(model).__name__
    try:
        name = model.name
        motif_length = model.motif_length
        left_context = model.left_context
        right_context = model.right_context
    except AttributeError as exc:
        raise ModelInterfaceError(
            capability, model_type, f"model is missing a required attribute: {exc}."
        ) from exc
    if not isinstance(name, str) or not name:
        raise ModelInterfaceError(
            capability, model_type, "name must be a non-empty string."
        )
    values = (
        ("motif_length", motif_length),
        ("left_context", left_context),
        ("right_context", right_context),
    )
    for field, value in values:
        try:
            integer = strict_integer(value, field)
        except TypeError:
            raise ModelInterfaceError(
                capability, model_type, f"{field} must be an integer."
            )
        if integer < 0 or (field == "motif_length" and integer == 0):
            requirement = "positive" if field == "motif_length" else "non-negative"
            raise ModelInterfaceError(
                capability, model_type, f"{field} must be {requirement}."
            )
    if not callable(getattr(model, "score_sites", None)):
        raise ModelInterfaceError(
            capability, model_type, "score_sites must be callable."
        )


@dataclass(frozen=True, slots=True)
class PWM(MotifModel):
    name: str
    weights: np.ndarray
    background: tuple
    sequence_scanner = True

    def __post_init__(self):
        _validate_model_name(self.name, "PWM")
        weights = _as_float32_readonly(self.weights)
        background = tuple(float(b) for b in self.background)
        _validate_pwm_weights(weights, background)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "background", background)

    @property
    def motif_length(self):
        return self.weights.shape[1]

    def score_sites(self, windows, /):
        from .backend import window_site_scores

        return window_site_scores(self, windows)

@dataclass(frozen=True, slots=True)
class _ContextModel(MotifModel):
    name: str
    weights: np.ndarray
    order: int
    motif_length: int
    sequence_scanner = True

    def __post_init__(self):
        _validate_model_name(self.name, type(self).__name__)
        weights = _as_float32_readonly(self.weights)
        order, motif_length = _validate_context_model(
            weights, self.order, self.motif_length, type(self).__name__, "order"
        )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "motif_length", motif_length)

    @property
    def left_context(self):
        return self.order

    @property
    def right_context(self):
        return self.order

    def score_sites(self, windows, /):
        from .backend import window_site_scores

        return window_site_scores(self, windows)


class BaMM(_ContextModel):
    pass


class Dimont(_ContextModel):
    pass


class Slim(_ContextModel):
    pass


@dataclass(frozen=True, slots=True)
class SiteGA(MotifModel):
    name: str
    weights: np.ndarray
    motif_length: int

    def __post_init__(self):
        _require_sitega_enabled()
        _validate_model_name(self.name, "SiteGA")
        weights = _as_float32_readonly(self.weights)
        motif_length = _validate_sitega(weights, self.motif_length)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "motif_length", motif_length)

    def score_sites(self, windows, /):
        """Dinucleotide scoring; equivalent to the dormant legacy reference."""
        weights = jnp.asarray(self.weights)
        total = jnp.zeros(windows.shape[0], dtype=jnp.float32)
        for term in range(self.motif_length - 1):
            code = 5 * windows[:, term].astype(jnp.int32) + windows[:, term + 1].astype(jnp.int32)
            total = total + weights[code, term]
        return total



# ── Geometry ─────────────────────────────────────────────────────────────────

def window_size(model):
    return (
        strict_integer(model.left_context, "left_context")
        + strict_integer(model.motif_length, "motif_length")
        + strict_integer(model.right_context, "right_context")
    )


def n_positions(model, sequence_length):
    sequence_length = strict_integer(sequence_length, "sequence_length")
    if sequence_length < 0:
        raise ValueError("sequence length must be non-negative.")
    width = window_size(model)
    if width < 1:
        raise ValueError("motif width must be positive.")
    return max(sequence_length - width + 1, 0)


def site_start_offset(model):
    return strict_integer(model.left_context, "left_context")


# ── PFM/PWM conversion ────────────────────────────────────────────────────────

def pfm_to_pwm(pfm, background=0.25):
    pfm = np.asarray(pfm, dtype=np.float32)
    if pfm.ndim != 2:
        raise ModelDimensionError(
            f"PFM must be two-dimensional, got {pfm.ndim} dimensions."
        )
    if pfm.shape[0] != NUCLEOTIDE_CARDINALITY:
        raise ModelDimensionError(f"PFM must have 4 rows, got {pfm.shape[0]}.")
    if pfm.shape[1] < 1:
        raise ModelDimensionError("PFM must contain at least one position.")
    if not np.all(np.isfinite(pfm)):
        raise ModelFormatError("", "PFM contains non-finite values.")
    if np.any((pfm < 0) | (pfm > 1)):
        raise ModelFormatError("", "PFM values must lie in [0, 1].")
    col_sums = pfm.sum(axis=0, dtype=np.float64)
    if not np.all(np.isclose(col_sums, 1.0, rtol=1e-4, atol=1e-6)):
        raise ModelFormatError("", "PFM columns must sum approximately to 1.0.")
    if not (math.isfinite(background) and background > 0):
        raise ModelFormatError("", "PWM background must be finite and positive.")
    if not math.isclose(4.0 * background, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelFormatError(
            "", f"PWM background sum is {4.0 * background}, expected approximately 1.0."
        )
    pc = np.float32(PSEUDOCOUNT_PWM)
    bg = np.float32(background)
    return np.log((pfm + pc) / bg).astype(np.float32)


def extend_pwm_with_n(weights4):
    weights4 = np.asarray(weights4, dtype=np.float32)
    if weights4.ndim != 2:
        raise ModelDimensionError(
            f"PWM weights must be two-dimensional, got {weights4.ndim} dimensions."
        )
    if weights4.shape[0] != NUCLEOTIDE_CARDINALITY:
        raise ModelDimensionError(
            f"PWM weights must have 4 rows to extend, got {weights4.shape[0]}."
        )
    n_row = weights4.min(axis=0, keepdims=True)
    return np.vstack([weights4, n_row]).astype(np.float32)


def pwm_from_pfm(pfm, background=0.25, name=""):
    _validate_model_name(name, "PWM")
    pwm4 = pfm_to_pwm(pfm, background=background)
    weights = extend_pwm_with_n(pwm4)
    bg = tuple(float(background) for _ in range(4))
    return PWM(name, weights, bg)
