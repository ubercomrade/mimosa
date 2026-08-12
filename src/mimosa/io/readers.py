"""Built-in model reader dispatch."""

from __future__ import annotations

from pathlib import Path

from ..errors import ModelFormatError, ModelInterfaceError
from ..models import MotifModel, _validate_model_contract, pwm_from_pfm
from .bundles import read_model_bundle
from .models import (
    parse_limited_xml,
    read_bamm,
    read_dimont,
    read_meme,
    read_pfm,
    read_sitega,
    read_slim,
)


def _normalise_format(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ModelFormatError(path, "model format must be a non-empty string.")
    return value.strip().lower().lstrip(".")


def _probe_xml(path, element_name):
    try:
        root = parse_limited_xml(path)
    except ModelFormatError:
        return False
    return root.find(f".//{element_name}") is not None


def _auto_format(path):
    if path.is_dir() and (path / "manifest.toml").is_file():
        return "bundle"
    suffix = path.suffix.lower()
    formats = {
        ".meme": "meme",
        ".pfm": "pfm",
        ".ihbcp": "bamm",
        ".mat": "sitega",
    }
    if suffix in formats:
        return formats[suffix]
    if suffix == ".xml":
        matches = [
            name
            for name, element in (("dimont", "MarkovModelDiffSM"), ("slim", "SLIM"))
            if _probe_xml(path, element)
        ]
        if len(matches) == 1:
            return matches[0]
    raise ModelFormatError(path, "cannot infer a unique built-in model format.")


def _read_builtin(path, format_name, index, background, order):
    if format_name == "bundle":
        return read_model_bundle(path)
    if format_name in ("meme", "pfm", "pwm"):
        source = read_meme if format_name in ("meme", "pwm") else read_pfm
        name, pfm = source(path, index=index) if source is read_meme else source(path)
        return pwm_from_pfm(pfm, background=background, name=name)
    if format_name == "bamm":
        return read_bamm(path, order=order)
    if format_name == "sitega":
        return read_sitega(path)
    if format_name == "dimont":
        return read_dimont(path)
    if format_name == "slim":
        return read_slim(path)
    raise ModelFormatError(path, f"unknown built-in model format '{format_name}'.")


def read_model(path, *, format="auto", index=0, background=0.25, order=None):
    """Read one built-in model or portable model bundle."""
    path = Path(path)
    format_name = _normalise_format(format, path)
    if format_name == "auto":
        format_name = _auto_format(path)
    elif format_name == "pwm" and path.suffix.lower() == ".pfm":
        format_name = "pfm"
    try:
        model = _read_builtin(path, format_name, index, background, order)
    except (ModelInterfaceError, AttributeError, OSError, TypeError, ValueError) as exc:
        raise ModelFormatError(path, f"reader failed: {exc}.") from exc
    if not isinstance(model, MotifModel):
        raise ModelFormatError(
            path, f"reader returned {type(model).__name__}, not a MotifModel."
        )
    try:
        _validate_model_contract(model, capability="read_model")
    except ModelInterfaceError as exc:
        raise ModelFormatError(path, exc.message) from exc
    return model
