"""Shared comparison helpers."""

from __future__ import annotations

from mimosa.cache import fingerprint_batch

ORIENTATION_TIEBREAK = {"++": 0, "+-": 1, "-+": 2, "--": 3}


def _select_best_orientation(candidates):
    """Choose the highest-scoring orientation with deterministic tie-breaking."""
    return max(
        candidates, key=lambda candidate: (float(candidate["score"]), -ORIENTATION_TIEBREAK[candidate["orientation"]])
    )


def _cached_batch_fingerprint(runtime_cache: dict, batch, label: str) -> str:
    runtime_key = ("batch_fp", label, id(batch))
    cached = runtime_cache.get(runtime_key)
    if cached is not None:
        return cached
    value = fingerprint_batch(batch) or f"no-{label}"
    runtime_cache[runtime_key] = value
    return value


def make_batch_preparation_context(sequences, background) -> dict:
    """Create the small shared, read-only part of a one-to-many runtime cache."""
    context: dict = {}
    _cached_batch_fingerprint(context, sequences, "sequences")
    _cached_batch_fingerprint(context, background, "background")
    return context
