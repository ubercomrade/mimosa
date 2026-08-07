"""Portable bundle storage: TOML manifest + raw Float32 blobs (models) and NPY (nulls).

Reproduces the Julia v2 model bundle and v7 null bundle layouts byte-for-byte,
including the `bitstring` fingerprint canonicalization.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import tempfile
import tomllib
from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from ..errors import InvariantError, ModelFormatError
from ..models import BaMM, Dimont, PWM, SiteGA, Slim

MODEL_FORMAT_VERSION = 2
NULL_FORMAT_VERSION = 7
BUNDLE_MANIFEST_NAME = "manifest.toml"
BUNDLE_DATA_DIR = "data"

MAX_BUNDLE_MANIFEST_BYTES = 134_217_728
MAX_BUNDLE_BLOB_BYTES = 1_073_741_824
MAX_BUNDLE_ARRAYS = 64
MAX_BUNDLE_RANK = 8
MAX_BUNDLE_DIMENSION = 100_000_000
MAX_BUNDLE_ELEMENTS = 100_000_000
MAX_BUNDLE_ALLOCATION_BYTES = 1_073_741_824


def _bundle_error(path, message):
    return ModelFormatError(str(path), message)


def _required_manifest_string(table, key, path, context):
    if key not in table:
        raise _bundle_error(path, f"{context} is missing '{key}'.")
    value = table[key]
    if not isinstance(value, str):
        raise _bundle_error(path, f"{context} '{key}' must be a string.")
    if not value:
        raise _bundle_error(path, f"{context} '{key}' must not be empty.")
    return value


def _required_manifest_table(table, key, path, context):
    if key not in table:
        raise _bundle_error(path, f"{context} is missing '{key}'.")
    value = table[key]
    if not isinstance(value, dict):
        raise _bundle_error(path, f"{context} '{key}' must be a TOML table.")
    return value


def _required_manifest_bool(table, key, path, context):
    if key not in table:
        raise _bundle_error(path, f"{context} is missing '{key}'.")
    value = table[key]
    if not isinstance(value, bool):
        raise _bundle_error(path, f"{context} '{key}' must be a boolean.")
    return value


def _required_manifest_int(table, key, path, context, minimum=-(2**63), maximum=2**63 - 1):
    if key not in table:
        raise _bundle_error(path, f"{context} is missing '{key}'.")
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bundle_error(path, f"{context} '{key}' must be an integer.")
    if value < minimum:
        raise _bundle_error(path, f"{context} '{key}' must be >= {minimum}.")
    if value > maximum:
        raise _bundle_error(path, f"{context} '{key}' must be <= {maximum}.")
    return value


def _required_manifest_float(table, key, path, context):
    if key not in table:
        raise _bundle_error(path, f"{context} is missing '{key}'.")
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bundle_error(path, f"{context} '{key}' must be numeric.")
    converted = float(value)
    if not np.isfinite(converted):
        raise _bundle_error(path, f"{context} '{key}' must be finite.")
    return converted


def _required_manifest_floats(table, key, path, context, expected_length=None):
    if key not in table:
        raise _bundle_error(path, f"{context} is missing '{key}'.")
    value = table[key]
    if not isinstance(value, list):
        raise _bundle_error(path, f"{context} '{key}' must be an array.")
    if expected_length is not None and len(value) != expected_length:
        raise _bundle_error(path, f"{context} '{key}' must contain {expected_length} values.")
    result = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _bundle_error(path, f"{context} '{key}' value {i} must be numeric.")
        converted = float(item)
        if not np.isfinite(converted):
            raise _bundle_error(path, f"{context} '{key}' value {i} must be finite.")
        result.append(converted)
    return result


def _required_manifest_shape(value, path, context):
    if not isinstance(value, list):
        raise _bundle_error(path, f"{context} shape must be an array.")
    if len(value) > MAX_BUNDLE_RANK:
        raise _bundle_error(path, f"{context} rank exceeds limit {MAX_BUNDLE_RANK}.")
    shape = []
    for i, dim in enumerate(value):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise _bundle_error(path, f"{context} dimension {i} must be an integer.")
        if dim < 0:
            raise _bundle_error(path, f"{context} dimension {i} is negative.")
        if dim > MAX_BUNDLE_DIMENSION:
            raise _bundle_error(path, f"{context} dimension {i} exceeds limit {MAX_BUNDLE_DIMENSION}.")
        shape.append(dim)
    return shape


def _bundle_dtype_bytes(dtype, path, context):
    if dtype not in ("<f4", "<f8", "<u4"):
        raise _bundle_error(
            path, f"{context} has unsupported dtype '{dtype}'; expected '<f4', '<f8' or '<u4'."
        )
    return 8 if dtype == "<f8" else 4


def _bundle_shape_payload_bytes(shape, dtype, path, context):
    item_size = _bundle_dtype_bytes(dtype, path, context)
    elements = 1
    for i, dim in enumerate(shape):
        if dim < 0:
            raise _bundle_error(path, f"{context} dimension {i} is negative.")
        if dim > MAX_BUNDLE_DIMENSION:
            raise _bundle_error(path, f"{context} dimension {i} exceeds bundle limit.")
        if dim == 0:
            elements = 0
        elif elements != 0:
            if elements > MAX_BUNDLE_ELEMENTS // dim:
                raise _bundle_error(path, f"{context} exceeds the element allocation budget.")
            elements *= dim
    if elements > MAX_BUNDLE_ALLOCATION_BYTES // item_size:
        raise _bundle_error(path, f"{context} exceeds the byte allocation budget.")
    return elements * item_size


def _validate_bundle_checksum(checksum, path, context):
    if not (isinstance(checksum, str) and len(checksum) == 71 and checksum.startswith("sha256:")):
        raise _bundle_error(path, f"{context} checksum must match sha256:<64 lowercase hex>.")
    try:
        int(checksum[7:], 16)
    except ValueError:
        raise _bundle_error(path, f"{context} checksum must match sha256:<64 lowercase hex>.")


def _resolve_bundle_path(root, relative, path, context, require_exists=False):
    if not relative:
        raise _bundle_error(path, f"{context} path must not be empty.")
    if "\x00" in relative:
        raise _bundle_error(path, f"{context} path contains NUL.")
    if os.path.isabs(relative) or relative.startswith("/") or "\\" in relative:
        raise _bundle_error(path, f"{context} path must be relative: '{relative}'.")
    parts = relative.split("/")
    if any(not p for p in parts) or any(p in (".", "..") for p in parts):
        raise _bundle_error(path, f"{context} path contains traversal components.")
    candidate = os.path.join(root, relative)
    if os.path.islink(candidate):
        raise _bundle_error(path, f"{context} path must not be a symbolic link.")
    if require_exists and not os.path.isfile(candidate):
        raise _bundle_error(path, f"{context} file does not exist: '{relative}'.")
    return candidate


def _file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _parse_bundle_array(root, arrays, name, path):
    if name not in arrays:
        raise _bundle_error(path, f"bundle is missing array '{name}'.")
    raw = arrays[name]
    if not isinstance(raw, dict):
        raise _bundle_error(path, f"array '{name}' must be a TOML table.")
    file = _required_manifest_string(raw, "file", path, f"array '{name}'")
    dtype = _required_manifest_string(raw, "dtype", path, f"array '{name}'")
    if "shape" not in raw:
        raise _bundle_error(path, f"array '{name}' is missing 'shape'.")
    shape = _required_manifest_shape(raw["shape"], path, f"array '{name}'")
    checksum = _required_manifest_string(raw, "checksum", path, f"array '{name}'")
    _bundle_dtype_bytes(dtype, path, f"array '{name}'")
    _validate_bundle_checksum(checksum, path, f"array '{name}'")
    _resolve_bundle_path(root, file, path, f"array '{name}'")
    _bundle_shape_payload_bytes(shape, dtype, path, f"array '{name}'")
    return file, dtype, shape, checksum


def _read_bundle_manifest(root, expected_version, expected_kind=None):
    if not os.path.isdir(root):
        raise _bundle_error(root, "bundle root is not a directory.")
    manifest_file = os.path.join(root, BUNDLE_MANIFEST_NAME)
    if not os.path.isfile(manifest_file):
        raise _bundle_error(root, "bundle is missing manifest.toml.")
    if os.path.islink(manifest_file):
        raise _bundle_error(root, "manifest must not be a symbolic link.")
    size = os.path.getsize(manifest_file)
    if size > MAX_BUNDLE_MANIFEST_BYTES:
        raise _bundle_error(root, f"manifest exceeds size limit {MAX_BUNDLE_MANIFEST_BYTES} bytes.")
    try:
        with open(manifest_file, "rb") as f:
            manifest = tomllib.load(f)
    except Exception as e:
        raise _bundle_error(root, f"invalid TOML manifest: {e}.")
    if not isinstance(manifest, dict):
        raise _bundle_error(root, "manifest must be a TOML table.")
    format_ = _required_manifest_string(manifest, "format", root, "manifest")
    if format_ != "mimosa":
        raise _bundle_error(root, f"unknown bundle format '{format_}'.")
    version = _required_manifest_int(manifest, "format_version", root, "manifest", minimum=1, maximum=expected_version)
    if version != expected_version:
        raise _bundle_error(root, f"unsupported bundle format version {version}.")
    kind = _required_manifest_string(manifest, "kind", root, "manifest")
    if expected_kind is not None and kind != expected_kind:
        raise _bundle_error(root, f"expected kind '{expected_kind}', got '{kind}'.")
    arrays = _required_manifest_table(manifest, "arrays", root, "manifest")
    if len(arrays) > MAX_BUNDLE_ARRAYS:
        raise _bundle_error(root, "manifest contains too many arrays.")
    total_payload = 0
    for name in arrays:
        file, dtype, shape, checksum = _parse_bundle_array(root, arrays, name, root)
        payload = _bundle_shape_payload_bytes(shape, dtype, root, f"array '{name}'")
        if total_payload > MAX_BUNDLE_ALLOCATION_BYTES - payload:
            raise _bundle_error(root, "bundle exceeds the total allocation budget.")
        total_payload += payload
    return manifest


def _validate_bundle_array_checksum(root, spec, bundle_path):
    file, dtype, shape, checksum = spec
    file_path = _resolve_bundle_path(root, file, bundle_path, f"array '{file}'", require_exists=True)
    if os.path.getsize(file_path) > MAX_BUNDLE_BLOB_BYTES:
        raise _bundle_error(bundle_path, f"binary blob exceeds size limit {MAX_BUNDLE_BLOB_BYTES} bytes.")
    actual = _file_sha256(file_path)
    if actual != checksum[7:]:
        raise _bundle_error(bundle_path, f"checksum mismatch for '{file}'.")
    return file_path


def _read_raw_f32_2d(path, expected_shape, expected_bytes, root=None, expected_checksum=None):
    file_size = os.path.getsize(path)
    if file_size != expected_bytes:
        raise _bundle_error(path, f"raw payload length mismatch: expected {expected_bytes} bytes, got {file_size}.")
    if _bundle_shape_payload_bytes(expected_shape, "<f4", path, "raw model array") != expected_bytes:
        raise _bundle_error(path, "raw payload length disagrees with shape.")
    if expected_checksum is not None:
        actual = _file_sha256(path)
        if actual != expected_checksum[7:]:
            raise _bundle_error(path, "checksum changed while reading the payload.")
    data = np.fromfile(path, dtype="<f4")
    if data.size != expected_shape[0] * expected_shape[1]:
        raise _bundle_error(path, "raw payload length disagrees with shape.")
    data = data.reshape(expected_shape)
    if not np.all(np.isfinite(data)):
        raise _bundle_error(path, "raw model array contains non-finite values.")
    return data


def _write_raw_f32_2d(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise InvariantError("model value cannot be represented as Float32.")
    arr.astype("<f4", copy=False).tofile(path)


def _write_npy(path, arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in (np.float64, np.uint32):
        raise InvariantError(f"unsupported NPY dtype {arr.dtype}.")
    np.save(path, arr)


def _read_npy(path, expected_dtype, expected_shape):
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        raise _bundle_error(path, f"NPY read failed: {e}.")
    if not arr.flags["C_CONTIGUOUS"]:
        raise _bundle_error(path, "NPY fortran_order does not match the row-major bundle contract.")
    dtype = "<f8" if arr.dtype == np.float64 else ("<u4" if arr.dtype == np.uint32 else str(arr.dtype))
    if dtype != expected_dtype:
        raise _bundle_error(path, f"NPY dtype '{dtype}' does not match '{expected_dtype}'.")
    if list(arr.shape) != list(expected_shape):
        raise _bundle_error(path, "NPY shape does not match the manifest.")
    return arr


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise InvariantError(f"unsupported TOML value type {type(v).__name__}")


def _toml_dump(manifest):
    lines = []

    def emit(table, prefix):
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        for key in sorted(scalars):
            lines.append(f"{key} = {_toml_value(scalars[key])}")
        for key in sorted(tables):
            path = f"{prefix}{key}"
            lines.append(f"[{path}]")
            emit(tables[key], f"{path}.")

    emit(manifest, "")
    return "\n".join(lines) + "\n"


def _write_bundle_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_toml_dump(manifest))


def _with_bundle_write(path, writer):
    target = os.path.abspath(str(path))
    if os.path.exists(target):
        raise InvariantError(f"bundle target '{target}' already exists.")
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=f".{os.path.basename(target)}.mimosa-stage-", dir=parent)
    try:
        os.makedirs(os.path.join(stage, BUNDLE_DATA_DIR), exist_ok=True)
        writer(target, stage)
        os.rename(stage, target)
        return target
    except Exception as e:
        if isinstance(e, (InvariantError, ModelFormatError)):
            raise
        raise InvariantError(f"failed to write bundle '{target}': {e}.")
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)


# ── Content fingerprints (Julia bitstring canonicalization) ─────────────────

def _content_fingerprint(arr, dtype, value_format, bits_format, bit_width, type_name):
    arr = np.asarray(arr, dtype=dtype)
    shape = "".join(f"{dimension}," for dimension in arr.shape)
    flat = arr.ravel(order="F")
    parts = [f"{type_name}:{shape};"]
    for x in flat:
        bits = struct.unpack(bits_format, struct.pack(value_format, float(x)))[0]
        parts.append(f"{bits:0{bit_width}b};")
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def content_fingerprint_float64(arr):
    """Return Julia-compatible Float64 content fingerprint."""
    return _content_fingerprint(arr, np.float64, "<d", "<Q", 64, "Float64")


def content_fingerprint_float32(arr):
    """Return Julia-compatible Float32 content fingerprint."""
    return _content_fingerprint(arr, np.float32, "<f", "<I", 32, "Float32")


def content_fingerprint_int64(arr):
    flat = np.asarray(arr).ravel()
    parts = ["integer-vector|"]
    for x in flat:
        parts.append(f"Int64:{int(x)};")
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def content_fingerprint_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def content_fingerprint_string(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def model_fingerprint(model):
    """SHA-256 fingerprint matching Julia's content_fingerprint(model)."""
    if isinstance(model, PWM):
        body = (
            content_fingerprint_float32(model.weights)
            + "|"
            + ",".join(str(b) for b in model.background)
        )
        return content_fingerprint_string(
            f"PWM{{Float32, Matrix{{Float32}}, NTuple{{4, Float32}}}}|{model.name}|{body}"
        )
    if isinstance(model, BaMM):
        body = (
            content_fingerprint_float32(model.weights)
            + f"|order={model.order},ml={model.motif_length}"
        )
        if model.order != 0:
            body += ",geometry=symmetric-v1"
        return content_fingerprint_string(
            f"BaMM{{Float32, Matrix{{Float32}}}}|{model.name}|{body}"
        )
    if isinstance(model, SiteGA):
        body = content_fingerprint_float32(model.weights) + f"|ml={model.motif_length}"
        return content_fingerprint_string(
            f"SiteGA{{Float32, Matrix{{Float32}}}}|{model.name}|{body}"
        )
    if isinstance(model, Dimont):
        body = (
            content_fingerprint_float32(model.weights)
            + f"|span={model.order},ml={model.motif_length}"
        )
        if model.order != 0:
            body += ",geometry=symmetric-v1"
        return content_fingerprint_string(
            f"Dimont{{Float32, Matrix{{Float32}}}}|{model.name}|{body}"
        )
    if isinstance(model, Slim):
        body = (
            content_fingerprint_float32(model.weights)
            + f"|span={model.order},ml={model.motif_length}"
        )
        if model.order != 0:
            body += ",geometry=symmetric-v1"
        return content_fingerprint_string(
            f"Slim{{Float32, Matrix{{Float32}}}}|{model.name}|{body}"
        )
    fp = model.fingerprint()
    if fp is None:
        raise ValueError(
            f"no content fingerprint is defined for {type(model).__name__}; "
            "implement fingerprint() for cache/null capability."
        )
    return fp


def model_collection_fingerprint(models):
    fps = sorted(model_fingerprint(m) for m in models)
    return content_fingerprint_string("|".join(fps))


def sequence_fingerprint(batch):
    data_fp = content_fingerprint_bytes(batch.data)
    offsets_fp = content_fingerprint_int64(batch.offsets)
    return content_fingerprint_string(f"batch:{data_fp}|{offsets_fp}")


def score_profile_fingerprint(profile):
    data_fp = content_fingerprint_float32(profile.scores.data)
    offsets_fp = content_fingerprint_int64(profile.scores.offsets)
    return content_fingerprint_string(
        f"ScoreProfile|layout=ragged-column-major|dtype=Float32|{profile.name}|data={data_fp}|offsets={offsets_fp}"
    )


# ── Model bundle write/read ──────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class _BundleSpec:
    kind: str
    array_name: str
    array: np.ndarray
    metadata: tuple


@singledispatch
def _bundle_spec(model):
    raise InvariantError(f"unsupported model kind {type(model).__name__}.")


@_bundle_spec.register(PWM)
def _bundle_spec_pwm(model):
    return _BundleSpec("pwm", "weights", model.weights, (("background", list(model.background)),))


@_bundle_spec.register(BaMM)
def _bundle_spec_bamm(model):
    return _BundleSpec(
        "bamm",
        "representation",
        model.weights,
        (("order", model.order), ("motif_length", model.motif_length)),
    )


@_bundle_spec.register(SiteGA)
def _bundle_spec_sitega(model):
    return _BundleSpec(
        "sitega", "representation", model.weights, (("motif_length", model.motif_length),)
    )


@_bundle_spec.register(Dimont)
def _bundle_spec_dimont(model):
    return _BundleSpec(
        "dimont",
        "representation",
        model.weights,
        (("span", model.order), ("motif_length", model.motif_length)),
    )


@_bundle_spec.register(Slim)
def _bundle_spec_slim(model):
    return _BundleSpec(
        "slim",
        "representation",
        model.weights,
        (("span", model.order), ("motif_length", model.motif_length)),
    )


def write_model(path, model):
    spec = _bundle_spec(model)
    arr = spec.array
    arr_name = spec.array_name
    shape = [arr.shape[0], arr.shape[1]]
    byte_length = _bundle_shape_payload_bytes(shape, "<f4", str(path), "model array")
    if not np.all(np.isfinite(arr)):
        raise InvariantError("model array contains non-finite values.")

    def writer(target, stage):
        data_path = os.path.join(stage, BUNDLE_DATA_DIR, f"{arr_name}.bin")
        _write_raw_f32_2d(data_path, arr)
        checksum = _file_sha256(data_path)
        manifest = {
            "format": "mimosa",
            "format_version": MODEL_FORMAT_VERSION,
            "kind": spec.kind,
            "name": model.name,
            "dtype": "<f4",
            "shape": shape,
            "layout": "row_major",
            "convention": "axes: (base, position) for matrix models, (context_code, position) for higher-order",
            "provenance": {"tool": "Mimosa.jl", "version": "0.1.0"},
            "arrays": {
                arr_name: {
                    "file": f"data/{arr_name}.bin",
                    "dtype": "<f4",
                    "shape": shape,
                    "byte_length": byte_length,
                    "checksum": f"sha256:{checksum}",
                }
            },
        }
        manifest.update(dict(spec.metadata))
        _write_bundle_manifest(os.path.join(stage, BUNDLE_MANIFEST_NAME), manifest)

    _with_bundle_write(path, writer)


def _read_model_array(path, manifest, array_name):
    layout = _required_manifest_string(manifest, "layout", path, "manifest")
    if layout != "row_major":
        raise _bundle_error(path, f"unsupported array layout '{layout}'.")
    dtype = _required_manifest_string(manifest, "dtype", path, "manifest")
    if "shape" not in manifest:
        raise _bundle_error(path, "manifest is missing 'shape'.")
    shape = _required_manifest_shape(manifest["shape"], path, "manifest")
    arrays = _required_manifest_table(manifest, "arrays", path, "manifest")
    file, spec_dtype, spec_shape, checksum = _parse_bundle_array(path, arrays, array_name, path)
    if spec_dtype != dtype:
        raise _bundle_error(path, "manifest and array dtype declarations disagree.")
    if spec_shape != shape:
        raise _bundle_error(path, "manifest and array shape declarations disagree.")
    if dtype != "<f4":
        raise _bundle_error(path, "model arrays must use dtype '<f4'.")
    file_path = _validate_bundle_array_checksum(path, (file, spec_dtype, spec_shape, checksum), path)
    byte_length = _required_manifest_int(
        arrays[array_name], "byte_length", path, f"array '{array_name}'", minimum=1, maximum=MAX_BUNDLE_BLOB_BYTES
    )
    return _read_raw_f32_2d(file_path, shape, byte_length, root=path, expected_checksum=checksum)


def _validate_declared_model_shape(path, manifest, expected_rows, expected_cols, model_kind):
    if "shape" not in manifest:
        raise _bundle_error(path, "manifest is missing 'shape'.")
    shape = _required_manifest_shape(manifest["shape"], path, "manifest")
    if shape != [expected_rows, expected_cols]:
        raise _bundle_error(
            path, f"{model_kind} manifest shape does not match model constructor invariants."
        )
    return shape


def _read_pwm_bundle(path, manifest, name):
    if "shape" not in manifest:
        raise _bundle_error(path, "manifest is missing 'shape'.")
    declared = _required_manifest_shape(manifest["shape"], path, "manifest")
    if len(declared) != 2 or declared[0] != 5 or declared[1] <= 0:
        raise _bundle_error(path, "PWM manifest shape must be [5, positive motif_length].")
    _validate_declared_model_shape(path, manifest, 5, declared[1], "PWM")
    weights = _read_model_array(path, manifest, "weights")
    bg_values = _required_manifest_floats(manifest, "background", path, "PWM manifest", expected_length=4)
    bg = tuple(np.float32(v) for v in bg_values)
    if not all(np.isfinite(bg)):
        raise _bundle_error(path, "PWM background is not representable as Float32.")
    return PWM(name, weights, bg)


def _read_bamm_bundle(path, manifest, name):
    order = _required_manifest_int(manifest, "order", path, "BaMM manifest", minimum=0, maximum=10)
    motif_length = _required_manifest_int(manifest, "motif_length", path, "BaMM manifest", minimum=1, maximum=10_000)
    _validate_declared_model_shape(path, manifest, 5 ** (order + 1), motif_length, "BaMM")
    representation = _read_model_array(path, manifest, "representation")
    return BaMM(name, representation, order, motif_length)


def _read_sitega_bundle(path, manifest, name):
    motif_length = _required_manifest_int(manifest, "motif_length", path, "SiteGA manifest", minimum=1, maximum=10_000)
    _validate_declared_model_shape(path, manifest, 25, motif_length, "SiteGA")
    representation = _read_model_array(path, manifest, "representation")
    return SiteGA(name, representation, motif_length)


def _read_dimont_bundle(path, manifest, name):
    span = _required_manifest_int(manifest, "span", path, "Dimont manifest", minimum=0, maximum=10)
    motif_length = _required_manifest_int(manifest, "motif_length", path, "Dimont manifest", minimum=1, maximum=10_000)
    _validate_declared_model_shape(path, manifest, 5 ** (span + 1), motif_length, "Dimont")
    representation = _read_model_array(path, manifest, "representation")
    return Dimont(name, representation, span, motif_length)


def _read_slim_bundle(path, manifest, name):
    span = _required_manifest_int(manifest, "span", path, "Slim manifest", minimum=0, maximum=10)
    motif_length = _required_manifest_int(manifest, "motif_length", path, "Slim manifest", minimum=1, maximum=10_000)
    _validate_declared_model_shape(path, manifest, 5 ** (span + 1), motif_length, "Slim")
    representation = _read_model_array(path, manifest, "representation")
    return Slim(name, representation, span, motif_length)


def read_model_bundle(path):
    manifest = _read_bundle_manifest(path, MODEL_FORMAT_VERSION)
    kind = _required_manifest_string(manifest, "kind", path, "manifest")
    name = _required_manifest_string(manifest, "name", path, "manifest")
    if kind == "pwm":
        return _read_pwm_bundle(path, manifest, name)
    if kind == "bamm":
        return _read_bamm_bundle(path, manifest, name)
    if kind == "sitega":
        return _read_sitega_bundle(path, manifest, name)
    if kind == "dimont":
        return _read_dimont_bundle(path, manifest, name)
    if kind == "slim":
        return _read_slim_bundle(path, manifest, name)
    raise _bundle_error(path, f"unknown model kind '{kind}'.")


# ── Null bundle write/read ───────────────────────────────────────────────────

def write_null_bundle(path, dist):
    if dist.n_null != len(dist.raw_scores):
        raise InvariantError("null distribution n_null does not match raw_scores length.")
    if len(dist.pairs) != dist.n_null:
        raise InvariantError("null distribution pairs do not match n_null.")
    if dist.n_models < 2:
        raise InvariantError("null distribution requires at least two source models.")
    if not dist.model_type:
        raise InvariantError("null distribution model_type must not be empty.")
    if dist.seed < 0:
        raise InvariantError("null distribution seed must be non-negative.")
    if not dist.sampling_version:
        raise InvariantError("null distribution sampling_version must not be empty.")
    if not dist.strategy or not dist.metric:
        raise InvariantError("null distribution strategy/metric must not be empty.")
    if dist.strategy != "profile":
        raise InvariantError("only profile null distributions are supported.")
    if dist.metric not in ("co", "co_rowwise", "dice", "dice_rowwise", "cosine"):
        raise InvariantError(f"unsupported profile metric '{dist.metric}'.")
    if not np.all(np.isfinite(dist.raw_scores)):
        raise InvariantError("null distribution raw_scores contain non-finite values.")
    _bundle_shape_payload_bytes([len(dist.raw_scores)], "<f8", str(path), "raw_null_scores array")

    def writer(target, stage):
        npy_path = os.path.join(stage, BUNDLE_DATA_DIR, "raw_null_scores.npy")
        _write_npy(npy_path, np.asarray(dist.raw_scores, dtype=np.float64))
        checksum = _file_sha256(npy_path)
        labels = []
        label_indices = {}
        for pair in dist.pairs:
            for label in (pair[0], pair[1]):
                if label not in label_indices:
                    label_indices[label] = len(labels) + 1
                    labels.append(label)
        pair_indices = np.empty((dist.n_null, 2), dtype=np.uint32)
        for index, pair in enumerate(dist.pairs):
            pair_indices[index, 0] = label_indices[pair[0]]
            pair_indices[index, 1] = label_indices[pair[1]]
        pairs_path = os.path.join(stage, BUNDLE_DATA_DIR, "pair_indices.npy")
        _write_npy(pairs_path, pair_indices)
        pairs_checksum = _file_sha256(pairs_path)
        manifest = {
            "format": "mimosa",
            "format_version": NULL_FORMAT_VERSION,
            "kind": "null_distribution",
            "strategy": dist.strategy,
            "metric": dist.metric,
            "estimator": "empirical_upper_tail",
            "n_null": dist.n_null,
            "n_models": dist.n_models,
            "model_type": dist.model_type,
            "shuffle": dist.shuffle,
            "seed": dist.seed,
            "sampling_version": dist.sampling_version,
            "pair_labels": labels,
            "compatibility": {
                "format_version": NULL_FORMAT_VERSION,
                "strategy": dist.strategy,
                "metric": dist.metric,
                "sequence_fingerprint": dist.sequence_fingerprint,
                "background_fingerprint": dist.background_fingerprint,
                "model_collection_fingerprint": dist.model_collection_fingerprint or "none",
                "model_type": dist.model_type,
                "shuffle": dist.shuffle,
                "sampling_version": dist.sampling_version,
                "search_range": dist.contract["search_range"],
                "window_radius": dist.contract["window_radius"],
                "realign_window": dist.contract["realign_window"],
                "min_logerr": float(dist.contract["min_logerr"]),
                "normalization_version": dist.contract["normalization_version"],
                "alignment_version": dist.contract["alignment_version"],
                "raw_scores_fingerprint": dist.contract["raw_scores_fingerprint"],
            },
            "arrays": {
                "raw_null_scores": {
                    "file": "data/raw_null_scores.npy",
                    "dtype": "<f8",
                    "shape": [len(dist.raw_scores)],
                    "checksum": f"sha256:{checksum}",
                },
                "pair_indices": {
                    "file": "data/pair_indices.npy",
                    "dtype": "<u4",
                    "shape": [dist.n_null, 2],
                    "checksum": f"sha256:{pairs_checksum}",
                },
            },
        }
        _write_bundle_manifest(os.path.join(stage, BUNDLE_MANIFEST_NAME), manifest)

    _with_bundle_write(path, writer)


def read_null_bundle(path):
    manifest = _read_bundle_manifest(path, NULL_FORMAT_VERSION, expected_kind="null_distribution")
    strategy = _required_manifest_string(manifest, "strategy", path, "null manifest")
    if strategy != "profile":
        raise _bundle_error(path, f"unsupported null strategy '{strategy}'.")
    metric = _required_manifest_string(manifest, "metric", path, "null manifest")
    n_null = _required_manifest_int(manifest, "n_null", path, "null manifest", minimum=0, maximum=MAX_BUNDLE_ELEMENTS)
    n_models = _required_manifest_int(manifest, "n_models", path, "null manifest", minimum=2, maximum=MAX_BUNDLE_ELEMENTS)
    model_type = _required_manifest_string(manifest, "model_type", path, "null manifest")
    shuffle = _required_manifest_bool(manifest, "shuffle", path, "null manifest")
    seed = _required_manifest_int(manifest, "seed", path, "null manifest", minimum=0)
    sampling_version = _required_manifest_string(manifest, "sampling_version", path, "null manifest")
    estimator = _required_manifest_string(manifest, "estimator", path, "null manifest")
    if estimator != "empirical_upper_tail":
        raise _bundle_error(path, f"unsupported null estimator '{estimator}'.")

    arrays = _required_manifest_table(manifest, "arrays", path, "null manifest")
    file, dtype, shape, checksum = _parse_bundle_array(path, arrays, "raw_null_scores", path)
    if dtype != "<f8":
        raise _bundle_error(path, "raw_null_scores must use dtype '<f8'.")
    if shape != [n_null]:
        raise _bundle_error(path, "raw_null_scores shape does not match n_null.")
    npy_path = _validate_bundle_array_checksum(path, (file, dtype, shape, checksum), path)
    raw_scores = _read_npy(npy_path, "<f8", [n_null]).astype(np.float64).ravel()
    if not np.all(np.isfinite(raw_scores)):
        raise _bundle_error(path, "raw_null_scores contains non-finite values.")

    labels_raw = manifest.get("pair_labels")
    if not isinstance(labels_raw, list):
        raise _bundle_error(path, "null manifest 'pair_labels' must be an array.")
    labels = []
    for index, label in enumerate(labels_raw):
        if not isinstance(label, str) or not label:
            raise _bundle_error(path, f"pair label {index} must be a non-empty string.")
        labels.append(label)
    if not labels:
        raise _bundle_error(path, "null manifest 'pair_labels' must not be empty.")
    if len(set(labels)) != len(labels):
        raise _bundle_error(path, "null manifest 'pair_labels' must be unique.")

    file, dtype, shape, checksum = _parse_bundle_array(path, arrays, "pair_indices", path)
    if dtype != "<u4":
        raise _bundle_error(path, "pair_indices must use dtype '<u4'.")
    if shape != [n_null, 2]:
        raise _bundle_error(path, "pair_indices shape does not match n_null.")
    indices_path = _validate_bundle_array_checksum(path, (file, dtype, shape, checksum), path)
    indices = _read_npy(indices_path, "<u4", [n_null, 2]).astype(np.int64)
    pairs = []
    for index in range(n_null):
        query_index = int(indices[index, 0]) - 1
        target_index = int(indices[index, 1]) - 1
        if not (0 <= query_index < len(labels)) or not (0 <= target_index < len(labels)):
            raise _bundle_error(path, f"pair index {index} is out of range.")
        pairs.append((labels[query_index], labels[target_index], float(raw_scores[index])))

    compat = _required_manifest_table(manifest, "compatibility", path, "null manifest")
    compat_version = _required_manifest_int(compat, "format_version", path, "compatibility metadata", minimum=NULL_FORMAT_VERSION, maximum=NULL_FORMAT_VERSION)
    if compat_version != NULL_FORMAT_VERSION:
        raise _bundle_error(path, "unsupported compatibility metadata version.")
    compat_strategy = _required_manifest_string(compat, "strategy", path, "compatibility metadata")
    compat_metric = _required_manifest_string(compat, "metric", path, "compatibility metadata")
    if compat_strategy != strategy:
        raise _bundle_error(path, "compatibility strategy disagrees with null manifest.")
    if compat_metric != metric:
        raise _bundle_error(path, "compatibility metric disagrees with null manifest.")
    seq_fp = _required_manifest_string(compat, "sequence_fingerprint", path, "compatibility metadata")
    bg_fp = _required_manifest_string(compat, "background_fingerprint", path, "compatibility metadata")
    mcf = _required_manifest_string(compat, "model_collection_fingerprint", path, "compatibility metadata")
    compat_model_type = _required_manifest_string(compat, "model_type", path, "compatibility metadata")
    if compat_model_type != model_type:
        raise _bundle_error(path, "compatibility model type disagrees with null manifest.")
    compat_shuffle = _required_manifest_bool(compat, "shuffle", path, "compatibility metadata")
    if compat_shuffle != shuffle:
        raise _bundle_error(path, "compatibility shuffle flag disagrees with null manifest.")
    compat_sampling = _required_manifest_string(compat, "sampling_version", path, "compatibility metadata")
    if compat_sampling != sampling_version:
        raise _bundle_error(path, "compatibility sampling version disagrees with null manifest.")
    search_range = _required_manifest_int(compat, "search_range", path, "compatibility metadata", minimum=0)
    window_radius = _required_manifest_int(compat, "window_radius", path, "compatibility metadata", minimum=0)
    realign_window = _required_manifest_int(compat, "realign_window", path, "compatibility metadata", minimum=0)
    min_logerr = _required_manifest_float(compat, "min_logerr", path, "compatibility metadata")
    normalization_version = _required_manifest_string(compat, "normalization_version", path, "compatibility metadata")
    alignment_version = _required_manifest_string(compat, "alignment_version", path, "compatibility metadata")
    raw_scores_fingerprint = _required_manifest_string(compat, "raw_scores_fingerprint", path, "compatibility metadata")
    actual_fp = content_fingerprint_float64(raw_scores)
    if raw_scores_fingerprint != actual_fp:
        raise _bundle_error(path, "raw score fingerprint does not match payload.")

    contract = {
        "metric": metric,
        "search_range": search_range,
        "window_radius": window_radius,
        "realign_window": realign_window,
        "min_logerr": np.float32(min_logerr),
        "normalization_version": normalization_version,
        "alignment_version": alignment_version,
        "sequence_fingerprint": seq_fp,
        "background_fingerprint": bg_fp,
        "raw_scores_fingerprint": raw_scores_fingerprint,
    }
    return {
        "strategy": strategy,
        "metric": metric,
        "raw_scores": raw_scores,
        "pairs": pairs,
        "n_null": n_null,
        "n_models": n_models,
        "model_type": model_type,
        "shuffle": shuffle,
        "seed": seed,
        "sampling_version": sampling_version,
        "model_collection_fingerprint": mcf if mcf != "none" else None,
        "sequence_fingerprint": seq_fp,
        "background_fingerprint": bg_fp,
        "contract": contract,
    }
