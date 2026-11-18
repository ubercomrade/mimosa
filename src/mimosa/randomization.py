"""Fixed-window scorers and coherent, position-specific alphabet relabeling.

An input-relabel surrogate evaluates f(pi(x)), using one fixed table per motif.
The single-window score distribution is invariant on uniform iid DNA; neither
overlapping-window dependence nor a biological null hypothesis is guaranteed.
"""

from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np

from .models import (
    PWM, BaMM, Dimont, MotifModel, SiteGA, Slim, DEFAULT_WINDOW_BATCH_SIZE,
    _require_sitega_enabled, _validate_model_contract, strict_integer, window_size,
)

INPUT_RELABEL_VERSION = "input-alphabet-relabel-v1"
INPUT_RELABEL_STRATEGY = "profile-input-relabel"


class WindowModel(MotifModel):
    """Adapt a deterministic batched function of oriented DNA windows.

    ``score_sites`` receives a JAX (n, width) uint8 array, with A/C/G/T/N encoded
    0/1/2/3/4, and returns n finite real scores. ``width`` includes both context
    flanks. Scores must not depend on batch size or other windows in the batch.
    The function must be JAX-traceable and return a real array.

    ``fingerprint`` identifies the function and learned parameters for caching
    and null construction. ``calibration_group`` identifies a common scoring
    formalism/configuration for null construction, not the identity of one TF.
    Changing either requires a different identifier. Neither is inferred from
    a Python callable's name or bytecode.
    """

    def __init__(
        self, name, motif_length, score_sites, *, left_context=0,
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
        if not callable(score_sites):
            raise TypeError("score_sites must be callable.")
        for field, value in (("fingerprint", fingerprint), ("calibration_group", calibration_group)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be a non-empty string when supplied.")
        self._score_sites = score_sites
        self._fingerprint = fingerprint
        self.calibration_group = calibration_group
        _validate_model_contract(self)

    def score_sites(self, windows):
        return self._score_sites(windows)

    def fingerprint(self):
        if self._fingerprint is None:
            return None
        identity = (
            f"window-model-v1|{self.name}|{self.motif_length}|{self.left_context}|"
            f"{self.right_context}|{self.calibration_group}|{self._fingerprint}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()


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
        transformed = jnp.asarray(self._full_permutations)[jnp.arange(windows.shape[1])[None, :], windows]
        return self.source_model.score_sites(transformed)

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
