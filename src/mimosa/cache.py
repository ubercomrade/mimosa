"""Optional prepared-profile cache: content-addressed, atomic writes."""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import tempfile
import tomllib
from contextlib import contextmanager
from threading import RLock

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np

from .io.bundles import (
    model_fingerprint,
    score_profile_fingerprint,
    sequence_fingerprint,
    toml_value,
)
from .profiles.anchors import AnchorCSR
from .profiles.normalization import (
    EmpiricalLogTail,
    HybridEmpiricalLogTail,
    normalization_fingerprint,
)
from .arrays import RaggedArray, StrandPair
from .profiles.prepared import PreparedProfile, ScoreProfile

CACHE_FORMAT_VERSION = 2
_CACHE_DATA_NAME = "data.bin"
_CACHE_META_NAME = "meta.toml"
_CACHE_LOCK_NAME = ".mimosa-cache.lock"

PREPARED_PROFILE_CACHE_FORMAT_VERSION = 4
_PREPARED_PROFILE_BINARY_MAGIC = b"MIMOSA-PREP-MMAP-1\0"
_PREPARED_PROFILE_SECTION_NAMES = (
    "forward_scores",
    "reverse_scores",
    "forward_score_offsets",
    "reverse_score_offsets",
    "forward_anchor_positions",
    "reverse_anchor_positions",
    "forward_anchor_offsets",
    "reverse_anchor_offsets",
)
class Cache:
    def __init__(self, directory):
        self.directory = str(directory)
        self._verified_entries = set()
        self._lock = RLock()

    def __repr__(self):
        return f"Cache({self.directory!r})"


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


def prepared_profile_cache_key(source, sequences=None, *, background=None, min_logerr=0.0, normalization=None):
    return _prepared_profile_cache_key(
        source,
        sequences,
        background=background,
        min_logerr=min_logerr,
        normalization=normalization,
    )


def _prepared_profile_cache_key(
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
    parts = (
        f"source={source_fingerprint}",
        sequence_part,
        background_part,
        f"min_logerr=0x{bits:08X}",
        f"normalization={normalization_fingerprint(normalization)}",
    )
    lines = [f"v={CACHE_FORMAT_VERSION}\n", "algo=prepared_profile\n", "algo_ver=4\n"]
    for part in parts:
        lines.extend((part, "\n"))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()[:16]


def _prepared_profile_sections(profile):
    forward = profile.bundle.forward
    reverse = profile.bundle.reverse
    forward_anchors, reverse_anchors = profile.anchors
    sections = {
        "forward_scores": (forward.data, "<f4"),
        "forward_score_offsets": (forward.offsets, "<i8"),
        "forward_anchor_positions": (forward_anchors.positions, "<i8"),
        "forward_anchor_offsets": (forward_anchors.offsets, "<i8"),
    }
    if reverse is forward:
        sections["reverse_scores"] = sections["forward_scores"]
        sections["reverse_score_offsets"] = sections["forward_score_offsets"]
    else:
        sections["reverse_scores"] = (reverse.data, "<f4")
        sections["reverse_score_offsets"] = (reverse.offsets, "<i8")
    if reverse_anchors is forward_anchors:
        sections["reverse_anchor_positions"] = sections["forward_anchor_positions"]
        sections["reverse_anchor_offsets"] = sections["forward_anchor_offsets"]
    else:
        sections["reverse_anchor_positions"] = (reverse_anchors.positions, "<i8")
        sections["reverse_anchor_offsets"] = (reverse_anchors.offsets, "<i8")
    return sections


def _encode_prepared_profile_with_metadata(profile):
    payload = bytearray(_PREPARED_PROFILE_BINARY_MAGIC)
    specs = {}
    written = {}
    sections = _prepared_profile_sections(profile)
    for name in _PREPARED_PROFILE_SECTION_NAMES:
        array, dtype = sections[name]
        identity = (id(array), dtype)
        if identity in written:
            specs[name] = written[identity]
            continue
        values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
        itemsize = values.dtype.itemsize
        offset = (len(payload) + itemsize - 1) // itemsize * itemsize
        payload.extend(b"\0" * (offset - len(payload)))
        raw = values.astype(dtype, copy=False).tobytes(order="C")
        payload.extend(raw)
        spec = {
            "offset": offset,
            "count": int(values.size),
            "dtype": dtype,
        }
        specs[name] = spec
        written[identity] = spec
    metadata = {
        "format": "prepared_profile_mmap",
        "algorithm": "prepared_profile",
        "prepared_profile_format_version": PREPARED_PROFILE_CACHE_FORMAT_VERSION,
        "name": profile.name,
        "min_logerr": float(profile.min_logerr),
        "normalization": normalization_fingerprint(profile.normalization),
        "n_rows": len(profile.bundle.forward),
        "shared_reverse_scores": profile.bundle.forward is profile.bundle.reverse,
        "shared_reverse_anchors": profile.anchors[0] is profile.anchors[1],
    }
    for name, spec in specs.items():
        metadata[f"{name}_offset"] = spec["offset"]
        metadata[f"{name}_count"] = spec["count"]
        metadata[f"{name}_dtype"] = spec["dtype"]
    return bytes(payload), metadata


def _metadata_checksum(meta):
    payload = "".join(
        f"{name} = {toml_value(meta[name])}\n"
        for name in sorted(meta)
        if name != "metadata_checksum"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache_metadata(cache, key):
    meta_path = _cache_file_path(cache, key, _CACHE_META_NAME)
    with open(meta_path, "rb") as f:
        meta = tomllib.load(f)
    expected = meta.get("checksum", "")
    if not (
        isinstance(expected, str)
        and len(expected) == 71
        and expected.startswith("sha256:")
    ):
        return None
    try:
        int(expected[7:], 16)
    except ValueError:
        return None
    metadata_checksum = meta.get("metadata_checksum")
    if metadata_checksum is None:
        return meta
    if not isinstance(metadata_checksum, str) or len(metadata_checksum) != 64 or metadata_checksum != _metadata_checksum(meta):
        return None
    return meta


def _verify_cache_data(cache, key, path, meta):
    expected = meta["checksum"][7:]
    size = os.path.getsize(path)
    declared_size = meta.get("size", size)
    if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size != size:
        return False
    verification_key = (key, expected, size)
    if verification_key not in cache._verified_entries:
        with open(path, "rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            return False
        cache._verified_entries.add(verification_key)
    return True


def _normalization_from_fingerprint(value):
    if value == "empirical-log-tail-v1":
        return EmpiricalLogTail()
    if not value.startswith("hybrid-log-tail-v2;"):
        return None
    fields = {}
    for part in value.split(";")[1:]:
        if "=" not in part:
            return None
        name, item = part.split("=", 1)
        fields[name] = item
    try:
        return HybridEmpiricalLogTail(int(fields["bins"]))
    except (KeyError, TypeError, ValueError):
        return None


def _mapped_profile_section(path, meta, name, expected_dtype, file_size):
    offset_key = f"{name}_offset"
    count_key = f"{name}_count"
    dtype_key = f"{name}_dtype"
    offset = meta.get(offset_key)
    count = meta.get(count_key)
    dtype = meta.get(dtype_key)
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or dtype != expected_dtype
    ):
        raise ValueError(f"invalid prepared-profile section '{name}'.")
    itemsize = np.dtype(expected_dtype).itemsize
    if offset < len(_PREPARED_PROFILE_BINARY_MAGIC) or offset % itemsize:
        raise ValueError(f"invalid prepared-profile section offset '{name}'.")
    section_size = count * itemsize
    if offset > file_size or section_size > file_size - offset:
        raise ValueError(f"prepared-profile section '{name}' exceeds payload.")
    if count == 0:
        values = np.empty(0, dtype=np.dtype(expected_dtype))
        values.setflags(write=False)
        return values
    return np.memmap(
        path,
        mode="r",
        dtype=np.dtype(expected_dtype),
        offset=offset,
        shape=(count,),
    )


def _decode_mmap_prepared_profile(path, meta):
    try:
        if meta.get("format") != "prepared_profile_mmap":
            return None
        if meta.get("prepared_profile_format_version") != PREPARED_PROFILE_CACHE_FORMAT_VERSION:
            return None
        name = meta.get("name")
        n_rows = meta.get("n_rows")
        threshold = meta.get("min_logerr")
        normalization_tag = meta.get("normalization")
        if (
            not isinstance(name, str)
            or isinstance(n_rows, bool)
            or not isinstance(n_rows, int)
            or n_rows < 0
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or not isinstance(normalization_tag, str)
        ):
            return None
        normalization = _normalization_from_fingerprint(normalization_tag)
        if normalization is None or normalization_fingerprint(normalization) != normalization_tag:
            return None
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            if f.read(len(_PREPARED_PROFILE_BINARY_MAGIC)) != _PREPARED_PROFILE_BINARY_MAGIC:
                return None

        forward_data = _mapped_profile_section(
            path, meta, "forward_scores", "<f4", file_size
        )
        reverse_data = (
            forward_data
            if meta.get("shared_reverse_scores")
            else _mapped_profile_section(path, meta, "reverse_scores", "<f4", file_size)
        )
        forward_offsets = _mapped_profile_section(
            path, meta, "forward_score_offsets", "<i8", file_size
        )
        reverse_offsets = (
            forward_offsets
            if meta.get("shared_reverse_scores")
            else _mapped_profile_section(
                path, meta, "reverse_score_offsets", "<i8", file_size
            )
        )
        forward_anchor_positions = _mapped_profile_section(
            path, meta, "forward_anchor_positions", "<i8", file_size
        )
        reverse_anchor_positions = (
            forward_anchor_positions
            if meta.get("shared_reverse_anchors")
            else _mapped_profile_section(
                path, meta, "reverse_anchor_positions", "<i8", file_size
            )
        )
        forward_anchor_offsets = _mapped_profile_section(
            path, meta, "forward_anchor_offsets", "<i8", file_size
        )
        reverse_anchor_offsets = (
            forward_anchor_offsets
            if meta.get("shared_reverse_anchors")
            else _mapped_profile_section(
                path, meta, "reverse_anchor_offsets", "<i8", file_size
            )
        )
        if (
            forward_offsets.size != n_rows + 1
            or reverse_offsets.size != n_rows + 1
            or forward_anchor_offsets.size != n_rows + 1
            or reverse_anchor_offsets.size != n_rows + 1
        ):
            return None
        forward = RaggedArray(forward_data, forward_offsets)
        reverse = (
            forward
            if meta.get("shared_reverse_scores")
            else RaggedArray(reverse_data, reverse_offsets)
        )
        if not np.all(np.isfinite(forward.data)) or not np.all(np.isfinite(reverse.data)):
            return None
        forward_anchor = AnchorCSR(forward_anchor_positions, forward_anchor_offsets)
        reverse_anchor = (
            forward_anchor
            if meta.get("shared_reverse_anchors")
            else AnchorCSR(reverse_anchor_positions, reverse_anchor_offsets)
        )
        anchors = (forward_anchor, reverse_anchor)
        return PreparedProfile(
            name,
            StrandPair(forward, reverse),
            anchors,
            np.float32(threshold),
            normalization,
        )
    except (OSError, TypeError, ValueError):
        return None


def _cached_mmap_prepared_profile(cache, key):
    path = _cache_file_path(cache, key, _CACHE_DATA_NAME)
    try:
        meta = _read_cache_metadata(cache, key)
        if meta is None or meta.get("format") != "prepared_profile_mmap":
            return None
        if not _verify_cache_data(cache, key, path, meta):
            return None
        return _decode_mmap_prepared_profile(path, meta)
    except (OSError, TypeError, ValueError):
        return None


def cache_get(cache, key):
    path = _cache_file_path(cache, key, _CACHE_DATA_NAME)
    try:
        meta = _read_cache_metadata(cache, key)
        if meta is None:
            return None
        with open(path, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != meta["checksum"][7:]:
            return None
        if "size" in meta and meta["size"] != len(data):
            return None
        return data
    except Exception:
        return None


def cache_set(cache, key, data, metadata=None):
    path = _cache_file_path(cache, key, _CACHE_DATA_NAME)
    data = bytes(data)
    checksum = hashlib.sha256(data).hexdigest()
    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "checksum": f"sha256:{checksum}",
        "size": len(data),
    }
    for name, value in (metadata or {}).items():
        if name not in ("format_version", "checksum", "size"):
            meta[name] = value
    meta["metadata_checksum"] = _metadata_checksum(meta)
    with cache._lock:
        root = _cache_root(cache)
        os.makedirs(root, exist_ok=True)
        # ponytail: one cache-wide lock; per-key locks only if write contention matters.
        with _cache_lock(root):
            with tempfile.TemporaryDirectory(
                prefix=".mimosa-cache-stage-", dir=root, ignore_cleanup_errors=True
            ) as stage:
                entry_stage = os.path.join(stage, _validate_cache_key(key))
                os.makedirs(entry_stage)
                with open(os.path.join(entry_stage, _CACHE_DATA_NAME), "wb") as f:
                    f.write(data)
                with open(os.path.join(entry_stage, _CACHE_META_NAME), "w", encoding="utf-8") as f:
                    for name in sorted(meta):
                        f.write(f"{name} = {toml_value(meta[name])}\n")
                target = _cache_entry_dir(cache, key)
                if os.path.exists(target):
                    shutil.rmtree(target)
                os.rename(entry_stage, target)
                cache._verified_entries = {
                    item for item in cache._verified_entries if item[0] != key
                }
                return path


def clearcache(cache):
    with cache._lock:
        root = _cache_root(cache)
        if not os.path.isdir(root):
            cache._verified_entries.clear()
            return 0
        with _cache_lock(root):
            cache._verified_entries.clear()
            count = 0
            for name in os.listdir(root):
                entry = os.path.join(root, name)
                if name.startswith(".mimosa-cache-stage-") or ".backup-" in name:
                    shutil.rmtree(entry, ignore_errors=True)
                    continue
                if os.path.isdir(entry) and not os.path.islink(entry):
                    try:
                        _validate_cache_key(name)
                        valid_metadata = _read_cache_metadata(cache, name) is not None
                    except (OSError, TypeError, ValueError):
                        valid_metadata = False
                    if valid_metadata and os.path.isfile(os.path.join(entry, _CACHE_DATA_NAME)):
                        shutil.rmtree(entry)
                        count += 1
            return count


def _make_preparation_context(sequences, background):
    sequence_fp = sequence_fingerprint(sequences)
    background_fp = (
        sequence_fp
        if background is None or background is sequences
        else sequence_fingerprint(background)
    )
    return sequence_fp, background_fp


def _cached_prepared_profile(
    cache,
    source,
    sequences,
    background,
    threshold,
    normalization,
    context=None,
):
    key = _prepared_profile_cache_key(
        source,
        sequences,
        background=background,
        min_logerr=threshold,
        normalization=normalization,
        sequence_fp=None if context is None else context[0],
        background_fp=None if context is None else context[1],
    )
    with cache._lock:
        profile = _cached_mmap_prepared_profile(cache, key)
        if profile is not None and profile.min_logerr == np.float32(threshold) and profile.normalization == normalization:
            return key, profile
        return key, None


def _store_prepared_profile(cache, key, profile):
    data, metadata = _encode_prepared_profile_with_metadata(profile)
    cache_set(
        cache,
        key,
        data,
        metadata=metadata,
    )
