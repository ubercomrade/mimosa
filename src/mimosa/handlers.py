"""Built-in motif model handlers and registrations."""

from __future__ import annotations

import os

import joblib
import numpy as np

from mimosa.batches import make_strand_bundle
from mimosa.functions import pfm_to_pwm
from mimosa.io import (
    parse_file_content,
    read_bamm,
    read_dimont,
    read_meme,
    read_pfm,
    read_scores,
    read_sitega,
    read_slim,
    write_pfm,
    write_sitega,
)
from mimosa.models import GenericModel, register_model_handler
from mimosa.scanning import (
    StrandMode,
    resolve_strand_mode,
    scan_with_batch_kernel,
    scan_with_batch_kernel_strands,
    score_bounds_from_model,
    score_bounds_from_representation,
)


def _load_pickled_generic_model(path: str, model_type: str) -> GenericModel:
    """Load a trusted joblib/pickle payload and validate its public model type."""
    model = joblib.load(path)
    if not isinstance(model, GenericModel):
        raise TypeError(f"Unsupported {model_type} pickle payload: expected GenericModel, got {type(model)!r}")
    return model


def _scan_pwm(model: GenericModel, sequences, strand: StrandMode):
    return scan_with_batch_kernel(model, sequences, strand, with_context=False)


def _scan_pwm_both(model: GenericModel, sequences):
    return scan_with_batch_kernel_strands(model, sequences, with_context=False)


def _write_pwm(model: GenericModel, path: str) -> None:
    pfm = model.config.get("_source_pfm")
    if pfm is None:
        raise ValueError("PWM serialization requires the source PFM in model.config['_source_pfm'].")
    write_pfm(np.asarray(pfm, dtype=np.float32), model.name, model.length, path)


def pwm_model_from_pfm(pfm: np.ndarray, name: str, length: int) -> GenericModel:
    """Build the internal PWM model representation from one PFM."""
    pwm = pfm_to_pwm(pfm)
    pwm_ext = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32, copy=False)
    return GenericModel("pwm", name, pwm_ext, int(length), {"kmer": 1, "_source_pfm": pfm})


def _load_pwm(path: str, kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())

    if ext == ".pkl":
        model = _load_pickled_generic_model(path, "PWM")
        if model.config.get("_source_pfm") is None:
            raise ValueError("Unsupported PWM pickle format: source PFM is missing from model.config['_source_pfm'].")
        return model

    if ext == ".meme":
        pfm, info, _ = read_meme(path, index=kwargs.get("index", 0))
        name, length = info
    elif ext == ".pfm":
        pfm, length = read_pfm(path)
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        raise ValueError(f"Unsupported PWM format: {path}")

    return pwm_model_from_pfm(pfm, name, int(length))


def _scan_sitega(model: GenericModel, sequences, strand: StrandMode):
    return scan_with_batch_kernel(model, sequences, strand, with_context=False)


def _scan_sitega_both(model: GenericModel, sequences):
    return scan_with_batch_kernel_strands(model, sequences, with_context=False)


def _write_sitega(model: GenericModel, path: str) -> None:
    write_sitega(model, path)


def _load_sitega(path: str, _kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())
    if ext == ".pkl":
        return _load_pickled_generic_model(path, "SiteGA")
    if ext != ".mat":
        raise ValueError(f"Unsupported SiteGA format: {path}")

    representation, name, length = read_sitega(path)
    representation = np.asarray(representation, dtype=np.float32)
    minimum, maximum = score_bounds_from_representation(representation)
    return GenericModel(
        "sitega",
        name,
        representation,
        int(length),
        {"kmer": 2, "minimum": float(minimum), "maximum": float(maximum)},
    )


def _sitega_score_bounds(model: GenericModel) -> tuple[float, float]:
    minimum = model.config.get("minimum")
    maximum = model.config.get("maximum")
    if minimum is not None and maximum is not None:
        return float(minimum), float(maximum)
    return score_bounds_from_model(model)


def _scan_bamm(model: GenericModel, sequences, strand: StrandMode):
    return scan_with_batch_kernel(model, sequences, strand, with_context=True)


def _scan_bamm_both(model: GenericModel, sequences):
    return scan_with_batch_kernel_strands(model, sequences, with_context=True)


def _load_bamm(path: str, kwargs: dict) -> GenericModel:
    if not path.endswith(".ihbcp") and not os.path.exists(path):
        ihbcp_path = f"{path}.ihbcp"
        if os.path.exists(ihbcp_path):
            path = ihbcp_path
        else:
            raise FileNotFoundError(f"BaMM file not found: {path}")

    _, max_order, _ = parse_file_content(path)
    target_order = kwargs.get("order")
    target_order = max_order if target_order is None else min(int(target_order), max_order)
    representation = read_bamm(path, target_order)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel(
        "bamm",
        name,
        np.asarray(representation, dtype=np.float32),
        representation.shape[-1],
        {"kmer": representation.ndim - 1, "order": target_order},
    )


def _dump_model(model: GenericModel, path: str) -> None:
    joblib.dump(model, path)


def _scan_dimont(model: GenericModel, sequences, strand: StrandMode):
    return scan_with_batch_kernel(model, sequences, strand, with_context=int(model.config.get("kmer", 1)) > 1)


def _scan_dimont_both(model: GenericModel, sequences):
    return scan_with_batch_kernel_strands(model, sequences, with_context=int(model.config.get("kmer", 1)) > 1)


def _load_dimont(path: str, _kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())
    if ext == ".pkl":
        return _load_pickled_generic_model(path, "Dimont")
    if ext != ".xml":
        raise ValueError(f"Unsupported Dimont format: {path}")

    representation, length, span = read_dimont(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel("dimont", name, np.asarray(representation, dtype=np.float32), length, {"kmer": span + 1})


def _scan_slim(model: GenericModel, sequences, strand: StrandMode):
    return scan_with_batch_kernel(model, sequences, strand, with_context=int(model.config.get("kmer", 1)) > 1)


def _scan_slim_both(model: GenericModel, sequences):
    return scan_with_batch_kernel_strands(model, sequences, with_context=int(model.config.get("kmer", 1)) > 1)


def _load_slim(path: str, _kwargs: dict) -> GenericModel:
    _, ext = os.path.splitext(path.lower())
    if ext == ".pkl":
        return _load_pickled_generic_model(path, "Slim")
    if ext != ".xml":
        raise ValueError(f"Unsupported Slim format: {path}")

    representation, length, span = read_slim(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel("slim", name, np.asarray(representation, dtype=np.float32), length, {"kmer": span + 1})


def _scan_scores(model: GenericModel, _sequences=None, _strand: StrandMode = "best"):
    scores = model.config["scores_data"]
    if resolve_strand_mode(_strand) == "both":
        return make_strand_bundle(scores, scores)
    return scores


def _scan_scores_both(model: GenericModel, _sequences=None):
    scores = model.config["scores_data"]
    return scores, scores


def _write_scores(_model: GenericModel, _path: str) -> None:
    raise NotImplementedError("Score profiles cannot be written to files")


def _scores_score_bounds(model: GenericModel) -> tuple[float, float]:
    values = model.config["scores_data"]["values"]
    mask = model.config["scores_data"]["mask"]
    if not np.any(mask):
        return 0.0, 0.0
    valid = values[mask]
    return float(np.min(valid)), float(np.max(valid))


def _load_scores(path: str, _kwargs: dict) -> GenericModel:
    scores_data = read_scores(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return GenericModel("scores", name, None, 0, {"scores_data": scores_data})


def register_builtin_handlers() -> None:
    """Register all built-in model handlers."""
    register_model_handler(
        "pwm",
        scan=_scan_pwm,
        scan_both=_scan_pwm_both,
        load=_load_pwm,
        write=_write_pwm,
        score_bounds=score_bounds_from_model,
    )
    register_model_handler(
        "sitega",
        scan=_scan_sitega,
        scan_both=_scan_sitega_both,
        load=_load_sitega,
        write=_write_sitega,
        score_bounds=_sitega_score_bounds,
    )
    register_model_handler(
        "bamm",
        scan=_scan_bamm,
        scan_both=_scan_bamm_both,
        load=_load_bamm,
        write=_dump_model,
        score_bounds=score_bounds_from_model,
    )
    register_model_handler(
        "dimont",
        scan=_scan_dimont,
        scan_both=_scan_dimont_both,
        load=_load_dimont,
        write=_dump_model,
        score_bounds=score_bounds_from_model,
    )
    register_model_handler(
        "slim",
        scan=_scan_slim,
        scan_both=_scan_slim_both,
        load=_load_slim,
        write=_dump_model,
        score_bounds=score_bounds_from_model,
    )
    register_model_handler(
        "scores",
        scan=_scan_scores,
        scan_both=_scan_scores_both,
        load=_load_scores,
        write=_write_scores,
        score_bounds=_scores_score_bounds,
    )


__all__ = ["pwm_model_from_pfm", "register_builtin_handlers"]
