"""Null-distribution metadata and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mimosa.cache import fingerprint_batch
from mimosa.types import ComparatorConfig

NULL_FORMAT_VERSION = 2


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON-compatible data with stable key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Return a content hash for one input file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata(  # noqa: PLR0913
    *,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
    model_collection_fingerprint: str | None = None,
    relation_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build the compatibility metadata stored in each null distribution file."""
    return {
        "format_version": NULL_FORMAT_VERSION,
        "strategy": strategy,
        "metric": config["metric"],
        "sequence_fingerprint": fingerprint_batch(sequences) or "none",
        "background_fingerprint": fingerprint_batch(background) or "none",
        "model_collection_fingerprint": model_collection_fingerprint,
        "relation_fingerprint": relation_fingerprint,
    }
