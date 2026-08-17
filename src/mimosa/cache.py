"""Optional prepared-profile cache: content-addressed, atomic writes."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
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
from .profiles.anchors import collect_both_anchors
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

PREPARED_PROFILE_CACHE_FORMAT_VERSION = 6
PREPARED_PROFILE_ALGORITHM_VERSION = 6
_PREPARED_PROFILE_BINARY_MAGIC = b"MIMOSA-PREP-MMAP-1\0"
_PREPARED_PROFILE_SECTION_NAMES = (
    "forward_scores",
    "reverse_scores",
    "forward_score_offsets",
    "reverse_score_offsets",
)
class Cache:
    def __init__(self, directory, *, timings=None):
        self.directory = str(directory)
        self._verified_entries = set()
        self._lock = RLock()
        self.timings = timings

    def __repr__(self):
        return f"Cache({self.directory!r})"


def _record_timing(cache, phase, elapsed):
    if cache.timings is not None:
        cache.timings[phase] = cache.timings.get(phase, 0.0) + elapsed


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
    from .models import MotifModel, site_start_offset

    is_motif = isinstance(source, MotifModel)
    if is_motif and sequences is None:
        raise ValueError("motif prepared-profile cache keys require comparison sequences.")
    if is_motif:
        source_fingerprint = model_fingerprint(source)
        source_site_start_offset = site_start_offset(source)
    elif isinstance(source, ScoreProfile):
        source_fingerprint = score_profile_fingerprint(source)
        source_site_start_offset = 0
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
    parts = (
        f"source={source_fingerprint}",
        sequence_part,
        background_part,
        f"normalization={normalization_fingerprint(normalization)}",
        f"site_start_offset={source_site_start_offset}",
    )
    lines = [
        f"v={CACHE_FORMAT_VERSION}\n",
        "algo=prepared_profile\n",
        f"algo_ver={PREPARED_PROFILE_ALGORITHM_VERSION}\n",
    ]
    for part in parts:
        lines.extend((part, "\n"))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()[:16]


def _prepared_profile_sections(bundle):
    forward = bundle.forward
    reverse = bundle.reverse
    sections = {
        "forward_scores": (forward.data, "<f4"),
        "forward_score_offsets": (forward.offsets, "<i8"),
    }
    if reverse is forward:
        sections["reverse_scores"] = sections["forward_scores"]
        sections["reverse_score_offsets"] = sections["forward_score_offsets"]
    else:
        sections["reverse_scores"] = (reverse.data, "<f4")
        sections["reverse_score_offsets"] = (reverse.offsets, "<i8")
    return sections


def _prepared_profile_serialization_plan(
    profile_name, bundle, normalization, site_start_offset
):
    payload_size = len(_PREPARED_PROFILE_BINARY_MAGIC)
    specs = {}
    written = {}
    sections = _prepared_profile_sections(bundle)
    writes = []
    for section_name in _PREPARED_PROFILE_SECTION_NAMES:
        array, dtype = sections[section_name]
        identity = (id(array), dtype)
        if identity in written:
            specs[section_name] = written[identity]
            continue
        values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
        itemsize = values.dtype.itemsize
        offset = (payload_size + itemsize - 1) // itemsize * itemsize
        payload_size = offset + values.nbytes
        spec = {
            "offset": offset,
            "count": int(values.size),
            "dtype": dtype,
        }
        specs[section_name] = spec
        written[identity] = spec
        writes.append((offset, values))
    metadata = {
        "format": "prepared_profile_mmap",
        "algorithm": "prepared_profile",
        "prepared_profile_format_version": PREPARED_PROFILE_CACHE_FORMAT_VERSION,
        "name": profile_name,
        "normalization": normalization_fingerprint(normalization),
        "site_start_offset": site_start_offset,
        "n_rows": len(bundle.forward),
        "shared_reverse_scores": bundle.forward is bundle.reverse,
    }
    for name, spec in specs.items():
        metadata[f"{name}_offset"] = spec["offset"]
        metadata[f"{name}_count"] = spec["count"]
        metadata[f"{name}_dtype"] = spec["dtype"]
    return metadata, writes, payload_size


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
    stat = os.stat(path)
    size = stat.st_size
    declared_size = meta.get("size", size)
    if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size != size:
        return False
    verification_key = (
        key,
        expected,
        size,
        stat.st_dev,
        stat.st_ino,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if verification_key not in cache._verified_entries:
        started = time.perf_counter()
        with open(path, "rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        _record_timing(cache, "cache_checksum", time.perf_counter() - started)
        if actual != expected:
            return False
        cache._verified_entries.add(verification_key)
    return True


def _normalization_from_fingerprint(value):
    if value == "empirical-log-tail-v1":
        return EmpiricalLogTail()
    if not value.startswith("hybrid-log-tail-v3;"):
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
        normalization_tag = meta.get("normalization")
        site_start_offset = meta.get("site_start_offset")
        if (
            not isinstance(name, str)
            or isinstance(n_rows, bool)
            or not isinstance(n_rows, int)
            or n_rows < 0
            or not isinstance(normalization_tag, str)
            or isinstance(site_start_offset, bool)
            or not isinstance(site_start_offset, int)
            or site_start_offset < 0
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
        if (
            forward_offsets.size != n_rows + 1
            or reverse_offsets.size != n_rows + 1
        ):
            return None
        forward = RaggedArray(forward_data, forward_offsets)
        reverse = (
            forward
            if meta.get("shared_reverse_scores")
            else RaggedArray(reverse_data, reverse_offsets)
        )
        return name, StrandPair(forward, reverse), normalization, site_start_offset
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
        started = time.perf_counter()
        decoded = _decode_mmap_prepared_profile(path, meta)
        _record_timing(cache, "cache_semantic_validation", time.perf_counter() - started)
        return decoded
    except (OSError, TypeError, ValueError):
        return None


def _cache_set_prepared_profile(
    cache, key, name, bundle, normalization, site_start_offset
):
    """Atomically stream normalized scores into one cache payload.

    The data sections are written directly to the staging file while the SHA-256
    is updated incrementally, avoiding an additional full-size Python payload.
    """
    encode_started = time.perf_counter()
    metadata, writes, payload_size = _prepared_profile_serialization_plan(
        name, bundle, normalization, site_start_offset
    )
    _record_timing(cache, "cache_encode", time.perf_counter() - encode_started)
    with cache._lock:
        root = _cache_root(cache)
        os.makedirs(root, exist_ok=True)
        lock_started = time.perf_counter()
        with _cache_lock(root):
            _record_timing(cache, "cache_lock_wait", time.perf_counter() - lock_started)
            with tempfile.TemporaryDirectory(
                prefix=".mimosa-cache-stage-", dir=root, ignore_cleanup_errors=True
            ) as stage:
                entry_stage = os.path.join(stage, _validate_cache_key(key))
                os.makedirs(entry_stage)
                data_path = os.path.join(entry_stage, _CACHE_DATA_NAME)
                checksum = hashlib.sha256()
                write_started = time.perf_counter()
                with open(data_path, "wb") as f:
                    cursor = 0
                    magic = _PREPARED_PROFILE_BINARY_MAGIC
                    f.write(magic)
                    checksum.update(magic)
                    cursor += len(magic)
                    for offset, values in writes:
                        padding = offset - cursor
                        if padding:
                            zeros = b"\0" * padding
                            f.write(zeros)
                            checksum.update(zeros)
                            cursor += padding
                        raw = memoryview(values).cast("B")
                        f.write(raw)
                        checksum.update(raw)
                        cursor += raw.nbytes
                    if cursor != payload_size:
                        raise RuntimeError("prepared-profile payload size mismatch.")
                _record_timing(cache, "cache_write", time.perf_counter() - write_started)
                meta = {
                    "format_version": CACHE_FORMAT_VERSION,
                    "checksum": f"sha256:{checksum.hexdigest()}",
                    "size": payload_size,
                    **metadata,
                }
                meta["metadata_checksum"] = _metadata_checksum(meta)
                with open(
                    os.path.join(entry_stage, _CACHE_META_NAME), "w", encoding="utf-8"
                ) as f:
                    for name in sorted(meta):
                        f.write(f"{name} = {toml_value(meta[name])}\n")
                target = _cache_entry_dir(cache, key)
                if os.path.exists(target):
                    shutil.rmtree(target)
                os.rename(entry_stage, target)
                cache._verified_entries = {
                    item for item in cache._verified_entries if item[0] != key
                }
                return os.path.join(target, _CACHE_DATA_NAME)


def clearcache(cache):
    with cache._lock:
        root = _cache_root(cache)
        home = os.path.abspath(os.path.expanduser("~"))
        if root == os.path.dirname(root) or root == home:
            raise ValueError("cache directory is too broad to clear.")
        if not os.path.isdir(root):
            if os.path.lexists(root):
                raise ValueError("cache directory must be a real directory, not a file or symlink.")
            cache._verified_entries.clear()
            return 0
        if os.path.islink(root):
            raise ValueError("cache directory must be a real directory, not a file or symlink.")
        with _cache_lock(root):
            cache._verified_entries.clear()
            count = 0
            for name in os.listdir(root):
                entry = os.path.join(root, name)
                if os.path.isdir(entry) and not os.path.islink(entry):
                    data_path = os.path.join(entry, _CACHE_DATA_NAME)
                    try:
                        _validate_cache_key(name)
                        metadata = _read_cache_metadata(cache, name)
                        valid_entry = (
                            metadata is not None
                            and os.path.isfile(data_path)
                            and not os.path.islink(data_path)
                            and _verify_cache_data(cache, name, data_path, metadata)
                        )
                    except (OSError, TypeError, ValueError):
                        valid_entry = False
                    if (
                        valid_entry
                    ):
                        shutil.rmtree(entry)
                        count += 1
            # Verification above can repopulate this set.  Removed entries
            # must never remain trusted if their key is reused later.
            cache._verified_entries.clear()
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
        cached = _cached_mmap_prepared_profile(cache, key)
    if cached is None:
        return key, None
    name, bundle, cached_normalization, site_start_offset = cached
    if cached_normalization != normalization:
        return key, None
    anchors = collect_both_anchors(
        bundle, threshold, position_offset=site_start_offset
    )
    return key, PreparedProfile(
        name,
        bundle,
        anchors,
        threshold,
        cached_normalization,
        site_start_offset,
    )


def _store_normalized_profile(
    cache, key, name, bundle, normalization, site_start_offset
):
    _cache_set_prepared_profile(
        cache, key, name, bundle, normalization, site_start_offset
    )
