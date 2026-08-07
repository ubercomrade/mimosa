"""Model-reader selection without a mutable plugin registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import xml.etree.ElementTree as ET

from ..errors import ModelFormatError, ModelInterfaceError
from ..models import MotifModel, _validate_model_contract, pwm_from_pfm
from .bundles import read_model_bundle
from .models import read_bamm, read_dimont, read_meme, read_pfm, read_sitega, read_slim


class ModelReader(Protocol):
    formats: tuple[str, ...]

    def probe(self, path: Path) -> bool: ...

    def read(self, path: Path, **kwargs) -> MotifModel: ...


@dataclass(frozen=True, slots=True)
class _BuiltinReader:
    formats: tuple[str, ...]
    suffixes: tuple[str, ...]
    kind: str

    def probe(self, path: Path) -> bool:
        if self.kind == "bundle":
            return path.is_dir() and (path / "manifest.toml").is_file()
        if path.suffix.lower() not in self.suffixes:
            return False
        if self.kind == "dimont":
            return _probe_xml(path, "MarkovModelDiffSM")
        if self.kind == "slim":
            return _probe_xml(path, "SLIM")
        return True

    def read(self, path: Path, **kwargs) -> MotifModel:
        if self.kind == "bundle":
            return read_model_bundle(path)
        if self.kind == "meme":
            name, pfm = read_meme(path, index=kwargs.get("index", 0))
            return pwm_from_pfm(
                pfm, background=kwargs.get("background", 0.25), name=name
            )
        if self.kind == "pfm":
            name, pfm = read_pfm(path)
            return pwm_from_pfm(
                pfm, background=kwargs.get("background", 0.25), name=name
            )
        if self.kind == "bamm":
            return read_bamm(path, order=kwargs.get("order"))
        if self.kind == "sitega":
            return read_sitega(path)
        if self.kind == "dimont":
            return read_dimont(path)
        if self.kind == "slim":
            return read_slim(path)
        raise ModelFormatError(path, f"unknown built-in model reader '{self.kind}'.")


_BUILTIN_READERS = (
    _BuiltinReader(("bundle",), (), "bundle"),
    _BuiltinReader(("meme",), (".meme",), "meme"),
    _BuiltinReader(("pfm",), (".pfm",), "pfm"),
    _BuiltinReader(("bamm",), (".ihbcp",), "bamm"),
    _BuiltinReader(("sitega",), (".mat",), "sitega"),
    _BuiltinReader(("dimont",), (".xml",), "dimont"),
    _BuiltinReader(("slim",), (".xml",), "slim"),
)


def _probe_xml(path, element_name):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return False
    return root.find(f".//{element_name}") is not None


def _normalise_format(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ModelFormatError(path, "model format must be a non-empty string.")
    return value.strip().lower().lstrip(".")


def _reader_formats(reader, path):
    formats = getattr(reader, "formats", None)
    if isinstance(formats, str):
        formats = (formats,)
    try:
        formats = tuple(_normalise_format(value, path) for value in formats)
    except TypeError as exc:
        raise ModelFormatError(path, "model reader formats must be an iterable of strings.") from exc
    if not formats:
        raise ModelFormatError(path, "model reader must declare at least one format.")
    if not callable(getattr(reader, "probe", None)) or not callable(
        getattr(reader, "read", None)
    ):
        raise ModelFormatError(path, "model reader must provide probe() and read().")
    return formats


def _reader_matches_suffix(reader, suffix, path):
    suffix = suffix.lower()
    if suffix == ".":
        return False
    if suffix in getattr(reader, "suffixes", ()):
        return True
    return suffix.lstrip(".") in _reader_formats(reader, path)


def _select_reader(path, format_name, external_readers):
    readers = (*_BUILTIN_READERS, *external_readers)
    if format_name != "auto":
        matches = [
            reader
            for reader in readers
            if format_name in _reader_formats(reader, path)
        ]
        if len(matches) != 1:
            detail = "no reader" if not matches else "multiple readers"
            raise ModelFormatError(
                path, f"format '{format_name}' has {detail}; expected exactly one."
            )
        return matches[0]

    suffix_matches = [
        reader
        for reader in readers
        if _reader_matches_suffix(reader, path.suffix, path)
    ]
    candidates = suffix_matches if suffix_matches else list(readers)
    if len(candidates) == 1 and suffix_matches:
        return candidates[0]
    matches = []
    for reader in candidates:
        try:
            if bool(reader.probe(path)):
                matches.append(reader)
        except (OSError, ValueError, TypeError):
            continue
    if len(matches) != 1:
        detail = "no reader" if not matches else "multiple readers"
        raise ModelFormatError(path, f"auto-detection found {detail}; expected exactly one.")
    return matches[0]


def read_model(
    path,
    *,
    format="auto",
    index=0,
    background=0.25,
    order=None,
    readers=(),
):
    """Read a built-in model or one supplied through an operation-local reader."""
    path = Path(path)
    format_name = _normalise_format(format, path)
    try:
        external_readers = tuple(readers)
    except TypeError as exc:
        raise ModelFormatError(path, "readers must be an iterable.") from exc

    if path.is_dir() and (path / "manifest.toml").is_file():
        reader = _BUILTIN_READERS[0]
    else:
        reader = _select_reader(path, format_name, external_readers)
    try:
        model = reader.read(
            path, index=index, background=background, order=order
        )
    except ModelFormatError:
        raise
    except (ModelInterfaceError, AttributeError, TypeError, ValueError) as exc:
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
