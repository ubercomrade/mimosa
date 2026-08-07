"""Optional prepared-profile cache: content-addressed, atomic writes."""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import tempfile
import tomllib

import numpy as np

from .arrays import RaggedArray, StrandPair
from .io.bundles import (
    model_fingerprint,
    score_profile_fingerprint,
    sequence_fingerprint,
)
from .profiles.anchors import AnchorCSR
from .profiles.normalization import (
    EmpiricalLogTail,
    HybridEmpiricalLogTail,
    normalization_fingerprint,
)
from .profiles.prepared import PreparedProfile, ScoreProfile

CACHE_FORMAT_VERSION = 2
PREPARED_PROFILE_CACHE_FORMAT_VERSION = 2
_PREPARED_PROFILE_CACHE_MAGIC = b"MIMOSA-PREP-1"
_CACHE_DATA_NAME = "data.bin"
_CACHE_META_NAME = "meta.toml"
_CACHE_LOCK_NAME = ".mimosa-cache.lock"

ALGORITHM_VERSIONS = {
    "pwm_scan": "1",
    "bamm_scan": "1",
    "sitega_scan": "1",
    "dimont_scan": "1",
    "slim_scan": "1",
    "motif_compare": "1",
    "profile_compare": "1",
    "prepared_profile": "2",
    "null_build": "1",
}


class Cache:
    def __init__(self, directory, enabled=True):
        self.directory = str(directory)
        self.enabled = bool(enabled)

    def __repr__(self):
        return f"Cache({self.directory!r}, enabled={self.enabled})"


def _validate_cache_key(key):
    if not key:
        raise ValueError("cache key must not be empty.")
    if key in (".", ".."):
        raise ValueError("cache key must be a single path component.")
    if "\x00" in key:
        raise ValueError("cache key must not contain NUL.")
    if os.path.isabs(key) or "/" in key or "\\" in key:
        raise ValueError("cache key must be a single path component.")
    if not all(c.isalnum() or c in "._-" for c in key) or len(key) > 128:
        raise ValueError("cache key must be 1-128 ASCII letters, digits, '.', '_' or '-'.")
    return key


def _cache_root(cache):
    return os.path.abspath(cache.directory)


def _cache_entry_dir(cache, key):
    value = _validate_cache_key(key)
    path = os.path.join(_cache_root(cache), value)
    if os.path.islink(path):
        raise ValueError("cache entry path must not be a symlink.")
    return path


def _cache_file_path(cache, key, name):
    path = os.path.join(_cache_entry_dir(cache, key), name)
    if os.path.islink(path):
        raise ValueError("cache file path must not be a symlink.")
    return path


def cache_key(cache, algorithm, *parts):
    algo_version = ALGORITHM_VERSIONS.get(algorithm, "0")
    lines = [f"v={CACHE_FORMAT_VERSION}\n", f"algo={algorithm}\n", f"algo_ver={algo_version}\n"]
    for p in parts:
        lines.append(p)
        lines.append("\n")
    full_hash = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return full_hash[:16]


def prepared_profile_cache_key(cache, source, sequences=None, *, background=None, min_logerr=0.0, normalization=None):
    from .models import MotifModel

    is_motif = isinstance(source, MotifModel)
    if is_motif and sequences is None:
        raise ValueError("motif prepared-profile cache keys require comparison sequences.")
    if is_motif:
        source_fingerprint = model_fingerprint(source)
    elif isinstance(source, ScoreProfile):
        source_fingerprint = score_profile_fingerprint(source)
    else:
        raise ValueError(f"unsupported cache source {type(source).__name__}.")
    sequence_part = "sequences=none" if sequences is None else f"sequences={sequence_fingerprint(sequences)}"
    effective_background = sequences if (is_motif and background is None) else background
    background_part = (
        "background=none"
        if effective_background is None
        else f"background={sequence_fingerprint(effective_background)}"
    )
    threshold = np.float32(min_logerr)
    if not np.isfinite(threshold):
        raise ValueError("min_logerr must be finite.")
    if normalization is None:
        normalization = HybridEmpiricalLogTail()
    bits = struct.unpack("<I", struct.pack("<f", threshold))[0]
    return cache_key(
        cache,
        "prepared_profile",
        f"source={source_fingerprint}",
        sequence_part,
        background_part,
        f"min_logerr=0x{bits:08X}",
        f"normalization={normalization_fingerprint(normalization)}",
    )


def _write_cache_u64(f, value):
    f.write(struct.pack("<Q", value))


def _read_cache_u64(f):
    data = f.read(8)
    if len(data) != 8:
        raise ValueError("truncated cached integer.")
    return struct.unpack("<Q", data)[0]


def _write_cache_string(f, value):
    data = value.encode("utf-8")
    _write_cache_u64(f, len(data))
    f.write(data)


def _read_cache_string(f):
    count = _read_cache_u64(f)
    data = f.read(count)
    if len(data) != count:
        raise ValueError("truncated cached string.")
    return data.decode("utf-8")


def _write_cache_f32(f, value):
    f.write(struct.pack("<f", value))


def _read_cache_f32(f):
    data = f.read(4)
    if len(data) != 4:
        raise ValueError("truncated cached float.")
    return struct.unpack("<f", data)[0]


def _write_cache_int_vector(f, values):
    values = np.asarray(values, dtype=np.int64)
    _write_cache_u64(f, values.size)
    f.write(values.astype("<i8").tobytes())


def _read_cache_int_vector(f):
    count = _read_cache_u64(f)
    data = f.read(8 * count)
    if len(data) != 8 * count:
        raise ValueError("truncated cached integer vector.")
    return np.frombuffer(data, dtype="<i8").copy()


def _write_cache_ragged(f, ragged):
    _write_cache_u64(f, ragged.data.size)
    f.write(ragged.data.astype("<f4").tobytes())
    _write_cache_int_vector(f, ragged.offsets)


def _read_cache_ragged(f):
    count = _read_cache_u64(f)
    data = f.read(4 * count)
    if len(data) != 4 * count:
        raise ValueError("truncated cached score vector.")
    values = np.frombuffer(data, dtype="<f4").copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("cached scores must be finite.")
    return RaggedArray(values, _read_cache_int_vector(f))


def _write_cache_anchor_csr(f, csr):
    _write_cache_int_vector(f, csr.positions)
    _write_cache_int_vector(f, csr.offsets)


def _read_cache_anchor_csr(f):
    return AnchorCSR(_read_cache_int_vector(f), _read_cache_int_vector(f))


def _encode_prepared_profile(profile):
    import io

    buf = io.BytesIO()
    buf.write(_PREPARED_PROFILE_CACHE_MAGIC)
    _write_cache_u64(buf, PREPARED_PROFILE_CACHE_FORMAT_VERSION)
    _write_cache_string(buf, profile.name)
    _write_cache_f32(buf, profile.min_logerr)
    _write_cache_string(buf, normalization_fingerprint(profile.normalization))
    _write_cache_ragged(buf, profile.bundle.forward)
    _write_cache_ragged(buf, profile.bundle.reverse)
    _write_cache_anchor_csr(buf, profile.anchors[0])
    _write_cache_anchor_csr(buf, profile.anchors[1])
    return buf.getvalue()


def _decode_prepared_profile(data):
    import io

    try:
        buf = io.BytesIO(data)
        if buf.read(len(_PREPARED_PROFILE_CACHE_MAGIC)) != _PREPARED_PROFILE_CACHE_MAGIC:
            return None
        if _read_cache_u64(buf) != PREPARED_PROFILE_CACHE_FORMAT_VERSION:
            return None
        name = _read_cache_string(buf)
        threshold = _read_cache_f32(buf)
        if not np.isfinite(threshold):
            return None
        tag = _read_cache_string(buf)
        if tag == "empirical-log-tail-v1":
            normalization = EmpiricalLogTail()
        elif tag.startswith("hybrid-log-tail-v2;"):
            fields = dict(part.split("=", 1) for part in tag.split(";")[1:])
            normalization = HybridEmpiricalLogTail(int(fields["bins"]))
        else:
            return None
        forward = _read_cache_ragged(buf)
        reverse = _read_cache_ragged(buf)
        anchors = (_read_cache_anchor_csr(buf), _read_cache_anchor_csr(buf))
        if buf.read(1):
            return None
        return PreparedProfile(name, StrandPair(forward, reverse), anchors, threshold, normalization)
    except Exception:
        return None


def cache_get(cache, key):
    if not cache.enabled:
        return None
    path = _cache_file_path(cache, key, _CACHE_DATA_NAME)
    meta_path = _cache_file_path(cache, key, _CACHE_META_NAME)
    try:
        with open(meta_path, "rb") as f:
            meta = tomllib.load(f)
        expected = meta.get("checksum", "")
        if not (isinstance(expected, str) and expected.startswith("sha256:")):
            return None
        with open(path, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != expected[7:]:
            return None
        return data
    except Exception:
        return None


def cache_set(cache, key, data, metadata=None):
    if not cache.enabled:
        return None
    path = _cache_file_path(cache, key, _CACHE_DATA_NAME)
    checksum = hashlib.sha256(data).hexdigest()
    meta = {"format_version": CACHE_FORMAT_VERSION, "checksum": f"sha256:{checksum}", "size": len(data)}
    for name, value in (metadata or {}).items():
        if name not in ("format_version", "checksum", "size"):
            meta[name] = value
    root = _cache_root(cache)
    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, _CACHE_LOCK_NAME)
    with open(lock_path, "a"):
        pass
    stage = tempfile.mkdtemp(prefix=".mimosa-cache-stage-", dir=root)
    entry_stage = os.path.join(stage, _validate_cache_key(key))
    os.makedirs(entry_stage)
    try:
        with open(os.path.join(entry_stage, _CACHE_DATA_NAME), "wb") as f:
            f.write(data)
        with open(os.path.join(entry_stage, _CACHE_META_NAME), "w", encoding="utf-8") as f:
            for k in sorted(meta):
                v = meta[k]
                if isinstance(v, str):
                    f.write(f'{k} = "{v}"\n')
                elif isinstance(v, bool):
                    f.write(f"{k} = {'true' if v else 'false'}\n")
                else:
                    f.write(f"{k} = {v}\n")
        target = _cache_entry_dir(cache, key)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.rename(entry_stage, target)
        return path
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def clearcache(cache, key=None):
    if not cache.enabled:
        return 0
    root = _cache_root(cache)
    if not os.path.isdir(root):
        return 0
    if key is not None:
        entry = _cache_entry_dir(cache, key)
        if os.path.isdir(entry):
            shutil.rmtree(entry)
            return 1
        return 0
    count = 0
    for name in os.listdir(root):
        entry = os.path.join(root, name)
        if name.startswith(".mimosa-cache-stage-") or ".backup-" in name:
            shutil.rmtree(entry, ignore_errors=True)
            continue
        if os.path.isdir(entry) and not os.path.islink(entry):
            if os.path.isfile(os.path.join(entry, _CACHE_DATA_NAME)) or os.path.isfile(
                os.path.join(entry, _CACHE_META_NAME)
            ):
                shutil.rmtree(entry)
                count += 1
    return count


def _cached_prepared_profile(cache, source, sequences, background, threshold, normalization):
    if cache is None or not cache.enabled:
        return None, None
    key = prepared_profile_cache_key(
        cache, source, sequences, background=background, min_logerr=threshold, normalization=normalization
    )
    data = cache_get(cache, key)
    if data is None:
        return key, None
    return key, _decode_prepared_profile(data)


def _store_prepared_profile(cache, key, profile):
    if cache is None or not cache.enabled or key is None:
        return profile
    cache_set(
        cache,
        key,
        _encode_prepared_profile(profile),
        metadata={
            "algorithm": "prepared_profile",
            "prepared_profile_format_version": PREPARED_PROFILE_CACHE_FORMAT_VERSION,
            "normalization": normalization_fingerprint(profile.normalization),
        },
    )
    return profile
