"""Null-distribution persistence helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import joblib

from mimosa.nulls.types import NullDistributionFile

NULL_CACHE_DIR = Path.home() / ".cache" / "mimosa" / "nulls"


def save_null_distribution_file(null_distribution_file: NullDistributionFile, path: str | Path) -> Path:
    """Persist one null distribution file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(null_distribution_file, output)
    return output


def install_null_distribution_file(path: str | Path, cache_dir: str | Path = NULL_CACHE_DIR) -> Path:
    """Copy one null distribution file into the user null-distribution cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / Path(path).name
    shutil.copy2(path, destination)
    return destination


def load_null_distribution_file(source: str | Path | dict[str, Any]) -> NullDistributionFile:
    """Load a trusted null distribution file from a path or return an in-memory payload."""
    if isinstance(source, dict):
        return cast(NullDistributionFile, source)
    return cast(NullDistributionFile, joblib.load(source))
