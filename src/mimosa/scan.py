"""Public scan API delegates to the model's own ``scan`` method."""

from __future__ import annotations

from .models import STRAND_POLICIES  # noqa: F401  (re-exported compatibility constant)

__all__ = ["scan", "_scan_batch", "STRAND_POLICIES"]


def _scan_batch(model, batch, strands, circular=False):
    return model.scan(batch, strands=strands, circular=circular)


def scan(model, sequences, *, strands="forward"):
    """Scan a motif model against an EncodedSequences batch.

    Returns a RaggedArray for forward/reverse/best, or a StrandPair for both.
    Built-in models and JAX-traceable custom window scorers use JAX. This is a
    thin wrapper around ``model.scan``; call the method directly for the
    circular layout or other strand policies.
    """
    return model.scan(sequences, strands=strands)
