"""Optional prepared-profile cache: content-addressed, atomic writes."""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import struct
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from collections import OrderedDict

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np

from .io.bundles import (
    model_fingerprint,
    score_profile_fingerprint,
    sequence_fingerprint,
)
from .profiles.normalization import (
    HybridEmpiricalLogTail,
    normalization_fingerprint,
)
from .profiles.prepared import PreparedProfile, ScoreProfile

CACHE_FORMAT_VERSION = 2
_CACHE_DATA_NAME = "data.bin"
_CACHE_META_NAME = "meta.toml"
_CACHE_LOCK_NAME = ".mimosa-cache.lock"

ALGORITHM_VERSIONS = {"prepared_profile": "3"}
_MEMORY_CACHE_MAX_PROFILES = 256


@dataclass(frozen=True)
class _PreparationContext:
    sequence_fingerprint: str
    background_fingerprint: str


class Cache:
    def __init__(self, directory, enabled=True):
        self.directory = str(directory)
        self.enabled = bool(enabled)
        self._prepared_profiles = OrderedDict()

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


@contextmanager
def _cache_lock(root):
    lock_path = os.path.join(root, _CACHE_LOCK_NAME)
    with open(lock_path, "a+b") as lock:
        if os.name == "nt":
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    return _prepared_profile_cache_key(
        cache,
        source,
        sequences,
        background=background,
        min_logerr=min_logerr,
        normalization=normalization,
    )


def _prepared_profile_cache_key(
    cache,
    source,
    sequences=None,
    *,
    background=None,
    min_logerr=0.0,
    normalization=None,
    sequence_fp=None,
    background_fp=None,
):
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
    if sequences is not None and sequence_fp is None:
        sequence_fp = sequence_fingerprint(sequences)
    sequence_part = "sequences=none" if sequence_fp is None else f"sequences={sequence_fp}"
    effective_background = sequences if (is_motif and background is None) else background
    if effective_background is None:
        background_part = "background=none"
    elif effective_background is sequences and sequence_fp is not None:
        background_part = f"background={sequence_fp}"
    elif background_fp is not None:
        background_part = f"background={background_fp}"
    else:
        background_part = f"background={sequence_fingerprint(effective_background)}"
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


def _encode_prepared_profile(profile):
    return pickle.dumps(profile, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_prepared_profile(data):
    try:
        profile = pickle.loads(data)
    except Exception:
        return None
    if not isinstance(profile, PreparedProfile):
        return None
    try:
        return PreparedProfile(
            profile.name,
            profile.bundle,
            profile.anchors,
            profile.min_logerr,
            profile.normalization,
        )
    except (TypeError, ValueError):
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


def cache_set(cache, key, data):
    if not cache.enabled:
        return None
    path = _cache_file_path(cache, key, _CACHE_DATA_NAME)
    checksum = hashlib.sha256(data).hexdigest()
    root = _cache_root(cache)
    os.makedirs(root, exist_ok=True)
    # ponytail: one cache-wide lock; per-key locks only if write contention matters.
    with _cache_lock(root):
        stage = tempfile.mkdtemp(prefix=".mimosa-cache-stage-", dir=root)
        entry_stage = os.path.join(stage, _validate_cache_key(key))
        os.makedirs(entry_stage)
        try:
            with open(os.path.join(entry_stage, _CACHE_DATA_NAME), "wb") as f:
                f.write(data)
            with open(os.path.join(entry_stage, _CACHE_META_NAME), "w", encoding="utf-8") as f:
                f.write(f'checksum = "sha256:{checksum}"\n')
            target = _cache_entry_dir(cache, key)
            if os.path.exists(target):
                shutil.rmtree(target)
            os.rename(entry_stage, target)
            return path
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def clearcache(cache):
    if not cache.enabled:
        return 0
    root = _cache_root(cache)
    if not os.path.isdir(root):
        cache._prepared_profiles.clear()
        return 0
    with _cache_lock(root):
        cache._prepared_profiles.clear()
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


def _make_preparation_context(sequences, background):
    if sequences is None:
        return None
    sequence_fp = sequence_fingerprint(sequences)
    background_fp = (
        sequence_fp
        if background is None or background is sequences
        else sequence_fingerprint(background)
    )
    return _PreparationContext(sequence_fp, background_fp)


def _memory_cache_get(cache, key):
    profile = cache._prepared_profiles.pop(key, None)
    if profile is not None:
        cache._prepared_profiles[key] = profile
    return profile


def _memory_cache_set(cache, key, profile):
    cache._prepared_profiles.pop(key, None)
    cache._prepared_profiles[key] = profile
    while len(cache._prepared_profiles) > _MEMORY_CACHE_MAX_PROFILES:
        cache._prepared_profiles.popitem(last=False)


def _cached_prepared_profile(
    cache,
    source,
    sequences,
    background,
    threshold,
    normalization,
    context=None,
):
    if cache is None or not cache.enabled:
        return None, None
    key = _prepared_profile_cache_key(
        cache,
        source,
        sequences,
        background=background,
        min_logerr=threshold,
        normalization=normalization,
        sequence_fp=None if context is None else context.sequence_fingerprint,
        background_fp=None if context is None else context.background_fingerprint,
    )
    cached = _memory_cache_get(cache, key)
    if cached is not None:
        return key, cached
    data = cache_get(cache, key)
    if data is None:
        return key, None
    profile = _decode_prepared_profile(data)
    if profile is not None:
        _memory_cache_set(cache, key, profile)
    return key, profile


def _store_prepared_profile(cache, key, profile):
    if cache is None or not cache.enabled or key is None:
        return profile
    cache_set(
        cache,
        key,
        _encode_prepared_profile(profile),
    )
    _memory_cache_set(cache, key, profile)
    return profile
