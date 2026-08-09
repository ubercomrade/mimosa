"""Motif model types: PWM, BaMM, SiteGA, Dimont, Slim, and the custom contract."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .errors import ModelDimensionError, ModelFormatError, ModelInterfaceError

NUCLEOTIDE_CARDINALITY = 4
PSEUDOCOUNT_PWM = 1e-4


def _validate_model_name(name, model_type):
    if not isinstance(name, str) or not name:
        raise ValueError(f"{model_type} name must be a non-empty string.")


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
    if context < 0:
        raise ModelDimensionError(
            f"{model_name} {context_name} must be non-negative, got {context}."
        )
    if context > 10:
        raise ModelDimensionError(
            f"{model_name} {context_name} must be <= 10 to avoid allocation blow-up, got {context}."
        )
    expected_rows = 5 ** (context + 1)
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


def _validate_sitega(representation, motif_length):
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


def _as_float32_readonly(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.setflags(write=False)
    return arr


class MotifModel(ABC):
    """Minimal custom-model contract: name, motif_length, scan_into."""

    name: str
    motif_length: int
    left_context: int = 0
    right_context: int = 0

    @abstractmethod
    def scan_into(self, sequence, forward, reverse, /):
        """Fill both strand score tracks for one validated sequence."""

    def fingerprint(self):
        return None


def _validate_model_contract(model, capability="scan"):
    """Validate the public model contract before any geometry is computed."""
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
        scanner = model.scan_into
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
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ModelInterfaceError(
                capability, model_type, f"{field} must be an integer."
            )
        if value < 0 or (field == "motif_length" and value == 0):
            requirement = "positive" if field == "motif_length" else "non-negative"
            raise ModelInterfaceError(
                capability, model_type, f"{field} must be {requirement}."
            )
    if not callable(scanner):
        raise ModelInterfaceError(
            capability, model_type, "scan_into must be callable."
        )


@dataclass(frozen=True, slots=True)
class PWM(MotifModel):
    name: str
    weights: np.ndarray
    background: tuple

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

    def scan_into(self, sequence, forward, reverse, /):
        from ._kernels import pwm_scan_forward, pwm_scan_reverse

        n_pos = forward.shape[0]
        pwm_scan_forward(self.weights, sequence, n_pos, self.motif_length, forward)
        pwm_scan_reverse(self.weights, sequence, n_pos, self.motif_length, reverse)


@dataclass(frozen=True, slots=True)
class _ContextModel(MotifModel):
    name: str
    weights: np.ndarray
    order: int
    motif_length: int

    def __post_init__(self):
        _validate_model_name(self.name, type(self).__name__)
        weights = _as_float32_readonly(self.weights)
        _validate_context_model(
            weights, self.order, self.motif_length, type(self).__name__, "order"
        )
        object.__setattr__(self, "weights", weights)

    @property
    def left_context(self):
        return self.order

    @property
    def right_context(self):
        return self.order

    def scan_into(self, sequence, forward, reverse, /):
        from ._kernels import rolling_scan_forward, rolling_scan_reverse

        n_pos = forward.shape[0]
        kmer = self.order + 1
        rolling_scan_forward(self.weights, sequence, kmer, self.motif_length, n_pos, forward)
        rolling_scan_reverse(self.weights, sequence, kmer, self.motif_length, n_pos, reverse)


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
        _validate_model_name(self.name, "SiteGA")
        weights = _as_float32_readonly(self.weights)
        _validate_sitega(weights, self.motif_length)
        object.__setattr__(self, "weights", weights)

    def scan_into(self, sequence, forward, reverse, /):
        from ._kernels import rolling_scan_forward, rolling_scan_reverse

        n_pos = forward.shape[0]
        rolling_scan_forward(self.weights, sequence, 2, self.motif_length - 1, n_pos, forward)
        rolling_scan_reverse(self.weights, sequence, 2, self.motif_length - 1, n_pos, reverse)



# ── Geometry ─────────────────────────────────────────────────────────────────

def window_size(model):
    return model.left_context + model.motif_length + model.right_context


def n_positions(model, sequence_length):
    if sequence_length < 0:
        raise ValueError("sequence length must be non-negative.")
    width = window_size(model)
    if width < 1:
        raise ValueError("motif width must be positive.")
    return max(sequence_length - width + 1, 0)


def site_start_offset(model):
    return model.left_context


# ── PFM/PWM conversion ────────────────────────────────────────────────────────

def pfm_to_pwm(pfm, background=0.25):
    pfm = np.asarray(pfm, dtype=np.float32)
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
