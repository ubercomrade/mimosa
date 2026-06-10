"""Formatting helpers."""

from __future__ import annotations


def format_params(params: dict) -> str:
    """Format parameters as a deterministic string key."""
    keys = sorted(params.keys())
    return "_".join(f"{key}-{params[key]}" for key in keys)
