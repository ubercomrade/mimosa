"""Experimental ordered-pair R-B Monte Carlo using only JAX score kernels.

Not a public API: shape-specialized compilation, resource budgeting and
cross-backend reproducibility still need validation.
"""

from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np

from ._jax_backend import compare_circular, normalize_exact, scan_circular
from .arrays import EncodedSequences
from .io.bundles import model_fingerprint
from .models import BaMM, Dimont, PWM, Slim, strict_integer
from .profiles.anchors import parse_min_logerr
from .randomization import relabel_model


def _permutations(model, seed, replicate, group):
    identity = model_fingerprint(model)
    digest = hashlib.sha256(f"mimosa-direct-mc-v1|{seed}|{replicate}|{identity}".encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))
    if group == "full":
        return np.array([rng.permutation(4) for _ in range(model.motif_length)], dtype=np.uint8)
    result = np.tile(np.arange(4, dtype=np.uint8), (model.motif_length, 1))
    for row in result:
        row[[0, 3]] = rng.permutation([0, 3])
        row[[1, 2]] = rng.permutation([1, 2])
    return result


def _profile(model, sequences, background):
    raw = jnp.concatenate(scan_circular(model, sequences))
    calibration = raw if background is None else jnp.concatenate(scan_circular(model, background))
    return normalize_exact(raw, calibration).reshape((2, sequences.data.size))


def direct_mc(queries, targets, sequences, *, k, seed, group="full",
              min_logerr=4.0, search_range=5, background=None):
    """Return observed Float32 scores, counts and p-values for ordered pairs."""
    k, seed = strict_integer(k, "k"), strict_integer(seed, "seed")
    search_range = strict_integer(search_range, "search_range")
    if k < 1 or seed < 0 or search_range < 0:
        raise ValueError("k must be positive; seed and search_range must be non-negative.")
    threshold = parse_min_logerr(min_logerr)
    if group not in ("full", "gc"):
        raise ValueError("group must be 'full' or 'gc'.")
    if not isinstance(sequences, EncodedSequences):
        raise TypeError("sequences must be an EncodedSequences batch.")
    if background is not None and not isinstance(background, EncodedSequences):
        raise TypeError("background must be an EncodedSequences batch.")
    queries, targets = list(queries), list(targets)
    if not all(isinstance(m, (PWM, BaMM, Dimont, Slim)) for m in queries + targets):
        raise TypeError("JAX Monte Carlo supports PWM, BaMM, Dimont and Slim only.")
    shape = (len(queries), len(targets))
    observed = np.empty(shape, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int64)
    query_profiles = [_profile(model, sequences, background) for model in queries]
    for j, target in enumerate(targets):
        target_profile = _profile(target, sequences, background)
        obs = [compare_circular(q, target_profile, sequences.offsets,
                                min_logerr=threshold, search_range=search_range)[0]
               for q in query_profiles]
        counter = [jnp.int64(0) for _ in query_profiles]
        for replicate in range(k):
            variant = relabel_model(
                target, permutations=_permutations(target, seed, replicate, group),
            )
            random_profile = _profile(variant, sequences, background)
            for i, query in enumerate(query_profiles):
                score = compare_circular(query, random_profile, sequences.offsets,
                                         min_logerr=threshold, search_range=search_range)[0]
                counter[i] = counter[i] + (score >= obs[i])
        if obs:
            observed[:, j] = np.asarray(jnp.stack(obs))
            counts[:, j] = np.asarray(jnp.stack(counter))
    return observed, counts, (counts + 1) / (k + 1)
