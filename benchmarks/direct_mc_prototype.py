"""Exploratory direct pairwise Monte Carlo reference; see direct_mc_protocol.md."""

import hashlib

import numpy as np

from mimosa import compare, prepare_profile, relabel_model
from mimosa.io.bundles import model_fingerprint
from mimosa.models import MotifModel, strict_integer


def permutations(model, seed, replicate, group):
    identity = model_fingerprint(model)
    digest = hashlib.sha256(f"mimosa-direct-mc-v1|{seed}|{replicate}|{identity}".encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))
    if group == "full":
        return np.array([rng.permutation(4) for _ in range(model.motif_length)], dtype=np.uint8)
    if group == "gc":
        result = np.tile(np.arange(4, dtype=np.uint8), (model.motif_length, 1))
        for row in result:
            row[[0, 3]] = rng.permutation([0, 3])
            row[[1, 2]] = rng.permutation([1, 2])
        return result
    raise ValueError("group must be 'full' or 'gc'")


def direct_mc(queries, targets, sequences, *, k, seed, group="full", min_logerr=4.0,
              search_range=5, background=None, normalization=None):
    """Return observed scores, exceedance counts and raw p-values for ordered pairs."""
    k = strict_integer(k, "k")
    seed = strict_integer(seed, "seed")
    if k < 1 or seed < 0:
        raise ValueError("k must be positive and seed non-negative")
    if group not in ("full", "gc"):
        raise ValueError("group must be 'full' or 'gc'")
    queries, targets = list(queries), list(targets)
    if not all(isinstance(m, MotifModel) for m in queries + targets):
        raise TypeError("all inputs must be motif models")
    def prepare(model):
        return prepare_profile(model, sequences, background=background, min_logerr=min_logerr,
                               normalization=normalization, circular=True)

    query_profiles = [prepare(m) for m in queries]
    observed = np.empty((len(queries), len(targets)), dtype=np.float32)
    counts = np.zeros(observed.shape, dtype=np.int64)
    for j, target in enumerate(targets):
        target_profile = prepare(target)
        for i, query_profile in enumerate(query_profiles):
            observed[i, j] = compare(query_profile, target_profile, search_range=search_range).score
        del target_profile
        for replicate in range(k):
            shuffled = relabel_model(target, permutations=permutations(target, seed, replicate, group))
            target_profile = prepare(shuffled)
            for i, query_profile in enumerate(query_profiles):
                score = compare(query_profile, target_profile, search_range=search_range).score
                counts[i, j] += score >= observed[i, j]
    return observed, counts, (counts + 1) / (k + 1)
