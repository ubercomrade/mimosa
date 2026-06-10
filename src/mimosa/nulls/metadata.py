"""Null-distribution metadata and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

from mimosa.cache import fingerprint_batch
from mimosa.types import ComparatorConfig

NULL_FORMAT_VERSION = 2
_SCORE_CONFIG_KEYS = (
    "metric",
    "search_range",
    "window_radius",
    "realign_window",
    "min_logfpr",
    "pfm_mode",
    "pfm_top_fraction",
    "profile_normalization",
)


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


def comparator_signature(config: ComparatorConfig) -> dict[str, Any]:
    """Extract score-affecting comparator options."""
    return {key: config.get(key) for key in _SCORE_CONFIG_KEYS if key in config}


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
    try:
        version = package_metadata.version("mimosa-tool")
    except package_metadata.PackageNotFoundError:
        version = "0+unknown"

    config_signature = comparator_signature(config)
    return {
        "format_version": NULL_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "metric": config["metric"],
        "config_signature": config_signature,
        "config_signature_hash": stable_hash(config_signature),
        "sequence_fingerprint": fingerprint_batch(sequences) or "none",
        "background_fingerprint": fingerprint_batch(background) or "none",
        "model_collection_fingerprint": model_collection_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "package_version": version,
    }
