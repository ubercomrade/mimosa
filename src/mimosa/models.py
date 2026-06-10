"""Model contracts and registry-backed model I/O."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mimosa.io import read_meme_many


@dataclass(eq=False)
class GenericModel:
    """Motif model container."""

    type_key: str
    name: str
    representation: Any
    length: int
    config: dict


@dataclass(frozen=True)
class ModelHandler:
    """Adapter bundle for one model family.

    The registry stores explicit handler objects rather than dictionaries so the
    extension contract is visible to users and type checkers.
    """

    scan: Callable[..., Any]
    scan_both: Callable[..., Any] | None
    load: Callable[..., GenericModel]
    write: Callable[[GenericModel, str], None]
    score_bounds: Callable[[GenericModel], tuple[float, float]]


registry: dict[str, ModelHandler] = {}


def register_model_handler(
    key: str,
    *,
    scan: Callable[..., Any],
    load: Callable[..., GenericModel],
    write: Callable[[GenericModel, str], None],
    score_bounds: Callable[[GenericModel], tuple[float, float]],
    scan_both: Callable[..., Any] | None = None,
) -> None:
    """Register one public model handler bundle."""
    registry[key] = ModelHandler(
        scan=scan,
        scan_both=scan_both,
        load=load,
        write=write,
        score_bounds=score_bounds,
    )


def get_model_handler(key: str) -> ModelHandler:
    """Return one registered handler bundle."""
    try:
        return registry[key]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Model strategy '{key}' not found. Available: {available}") from exc


def write_model(model: GenericModel, path: str) -> None:
    """Universal write function that dispatches to the appropriate handler."""
    handler = get_model_handler(model.type_key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handler.write(model, path)


def read_model(path: str, model_type: str, **kwargs) -> GenericModel:
    """Factory function for creating models from files."""
    handler = get_model_handler(model_type)
    return handler.load(path, kwargs)


def read_models(
    path: str | Path,
    model_type: str,
    pattern: str | None = None,
    *,
    allow_duplicate_names: bool = False,
    **kwargs,
) -> list[GenericModel]:
    """Read a deterministic motif collection from a directory or multi-motif MEME file."""
    source = Path(path)
    if source.is_dir():
        paths = sorted(p for p in source.glob(pattern or "*") if p.is_file())
        models = [read_model(str(item), model_type, **kwargs) for item in paths]
    elif model_type == "pwm" and source.suffix.lower() == ".meme":
        from mimosa.handlers import pwm_model_from_pfm

        models = [pwm_model_from_pfm(pfm, name, length) for pfm, (name, length) in read_meme_many(source)]
    else:
        raise ValueError("Collection loading requires a directory or a multi-motif MEME file with model_type='pwm'.")

    _validate_unique_model_names(models, allow_duplicate_names=allow_duplicate_names)
    return models


def _validate_unique_model_names(models: list[GenericModel], *, allow_duplicate_names: bool) -> None:
    if allow_duplicate_names:
        return

    seen: set[str] = set()
    duplicates: set[str] = set()
    for model in models:
        if model.name in seen:
            duplicates.add(model.name)
        seen.add(model.name)

    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate motif names are not allowed: {names}")


def _register_builtin_handlers_once() -> None:
    """Install built-in model handlers when this module is imported."""
    from mimosa.handlers import register_builtin_handlers

    register_builtin_handlers()


_register_builtin_handlers_once()


__all__ = [
    "GenericModel",
    "ModelHandler",
    "get_model_handler",
    "read_model",
    "read_models",
    "register_model_handler",
    "registry",
    "write_model",
]
