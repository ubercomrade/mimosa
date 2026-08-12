import os
import multiprocessing
import tomllib

import numpy as np
import pytest

from mimosa.cache import (
    Cache,
    cache_set,
    clearcache,
    prepared_profile_cache_key,
)
from mimosa.io.fasta import read_fasta
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm
from mimosa.profiles.prepared import prepare_profile


def _hold_cache_lock(directory, ready, release):
    from mimosa.cache import _cache_lock

    with _cache_lock(directory):
        ready.set()
        release.wait(5)


def _wait_for_cache_lock(directory, acquired):
    from mimosa.cache import _cache_lock

    with _cache_lock(directory):
        acquired.set()


class TestCache:
    def test_set_writes_checksum_protected_entry(self, tmp_path):
        cache = Cache(str(tmp_path))
        cache_set(cache, "abc123", b"payload")
        with open(tmp_path / "abc123" / "meta.toml", "rb") as f:
            metadata = tomllib.load(f)
        assert metadata["size"] == len(b"payload")
        assert metadata["checksum"].startswith("sha256:")

    def test_clear(self, tmp_path):
        cache = Cache(str(tmp_path))
        cache_set(cache, "a", b"1")
        cache_set(cache, "b", b"2")
        assert clearcache(cache) == 2
        assert not (tmp_path / "a").exists()
        assert not cache._verified_entries

    def test_clear_preserves_unrelated_directory(self, tmp_path):
        unrelated = tmp_path / "not-a-cache-entry"
        unrelated.mkdir()
        (unrelated / "data.bin").write_bytes(b"user data")
        assert clearcache(Cache(str(tmp_path))) == 0
        assert unrelated.exists()

    def test_clear_preserves_checksum_invalid_entry(self, tmp_path):
        cache = Cache(str(tmp_path))
        cache_set(cache, "damaged", b"payload")
        data_path = tmp_path / "damaged" / "data.bin"
        data_path.write_bytes(b"changed")

        assert clearcache(cache) == 0
        assert data_path.exists()

    @pytest.mark.parametrize(
        "name", ("user.backup-data", ".mimosa-cache-stage-user-data")
    )
    def test_clear_preserves_user_directories_that_match_old_cleanup_names(
        self, tmp_path, name
    ):
        user_directory = tmp_path / name
        user_directory.mkdir()
        user_file = user_directory / "important.txt"
        user_file.write_text("do not delete")

        assert clearcache(Cache(str(tmp_path))) == 0
        assert user_file.read_text() == "do not delete"

    def test_clear_rejects_symlinked_cache_root(self, tmp_path):
        target = tmp_path / "real-cache"
        cache_set(Cache(str(target)), "abc", b"payload")
        link = tmp_path / "cache-link"
        link.symlink_to(target, target_is_directory=True)

        with pytest.raises(ValueError, match="real directory"):
            clearcache(Cache(str(link)))
        assert (target / "abc").exists()

    def test_key_validation(self, tmp_path):
        cache = Cache(str(tmp_path))
        with pytest.raises(ValueError):
            cache_set(cache, "../escape", b"x")
        with pytest.raises(ValueError):
            cache_set(cache, "a/b", b"x")

    def test_lock_serializes_processes(self, tmp_path):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        acquired = context.Event()
        holder = context.Process(target=_hold_cache_lock, args=(str(tmp_path), ready, release))
        waiter = None
        holder.start()
        try:
            assert ready.wait(5)
            waiter = context.Process(target=_wait_for_cache_lock, args=(str(tmp_path), acquired))
            waiter.start()
            assert not acquired.wait(0.2)
            release.set()
            assert acquired.wait(5)
        finally:
            release.set()
            holder.join(5)
            if waiter is not None:
                waiter.join(5)
            if holder.is_alive():
                holder.terminate()
            if waiter is not None and waiter.is_alive():
                waiter.terminate()
        assert holder.exitcode == 0
        assert waiter is not None and waiter.exitcode == 0

class TestPreparedProfileCache:
    @pytest.fixture
    def pwm(self):
        name, pfm = read_meme("examples/foxa2.meme")
        return pwm_from_pfm(pfm, name=name)

    @pytest.fixture
    def batch(self):
        return read_fasta("examples/foreground.fa")[0]

    def test_key_content_addressed(self, pwm, batch):
        k1 = prepared_profile_cache_key(pwm, batch)
        k2 = prepared_profile_cache_key(pwm, batch)
        assert k1 == k2
        assert len(k1) == 16

    def test_key_fingerprints_same_sequences_once(self, pwm, batch, monkeypatch):
        from mimosa import cache as cache_module

        calls = 0
        original = cache_module.sequence_fingerprint

        def counted(value):
            nonlocal calls
            calls += 1
            return original(value)

        monkeypatch.setattr(cache_module, "sequence_fingerprint", counted)
        prepared_profile_cache_key(pwm, batch)
        assert calls == 1

    def test_key_changes_with_model(self, pwm, batch):
        k1 = prepared_profile_cache_key(pwm, batch)
        from mimosa import PWM

        m2 = PWM("other", pwm.weights, pwm.background)
        k2 = prepared_profile_cache_key(m2, batch)
        assert k1 != k2

    def test_key_changes_with_min_logerr(self, pwm, batch):
        k1 = prepared_profile_cache_key(pwm, batch, min_logerr=0.0)
        k2 = prepared_profile_cache_key(pwm, batch, min_logerr=2.0)
        assert k1 != k2

    def test_prepare_hit(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        p1 = prepare_profile(pwm, batch, cache=cache)
        p2 = prepare_profile(pwm, batch, cache=cache)
        assert p1 == p2

    def test_prepare_cache_has_no_in_memory_profile_store(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        prepare_profile(pwm, batch, cache=cache)
        assert not hasattr(cache, "_prepared_profiles")
        assert not hasattr(cache, "memory_budget_bytes")

    def test_prepare_disk_hit_maps_read_only_arrays(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        expected = prepare_profile(pwm, batch, cache=cache)
        loaded = prepare_profile(pwm, batch, cache=Cache(str(tmp_path)))
        assert loaded == expected
        assert not loaded.bundle.forward.data.flags.writeable
        assert not loaded.bundle.forward.offsets.flags.writeable
        assert loaded.bundle.forward.data.base is not None

    def test_prepare_disk_hit_rejects_corrupt_payload(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        expected = prepare_profile(pwm, batch, cache=cache)
        key = prepared_profile_cache_key(pwm, batch)
        data_path = tmp_path / key / "data.bin"
        with open(data_path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            value = f.read(1)[0]
            f.seek(-1, os.SEEK_END)
            f.write(bytes([value ^ 0xFF]))
        loaded = prepare_profile(pwm, batch, cache=Cache(str(tmp_path)))
        assert loaded == expected

    def test_verified_entry_rechecks_same_size_payload_mutation(
        self, pwm, batch, tmp_path, monkeypatch
    ):
        import hashlib

        cache = Cache(str(tmp_path))
        prepare_profile(pwm, batch, cache=cache)
        prepare_profile(pwm, batch, cache=cache)
        key = prepared_profile_cache_key(pwm, batch)
        data_path = tmp_path / key / "data.bin"
        with open(data_path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            value = f.read(1)[0]
            f.seek(-1, os.SEEK_END)
            f.write(bytes([value ^ 0xFF]))

        calls = 0
        original = hashlib.file_digest

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(hashlib, "file_digest", counted)
        prepare_profile(pwm, batch, cache=cache)
        assert calls == 1

    def test_prepare_payload_is_mmap_profile(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        prepared = prepare_profile(pwm, batch, cache=cache)
        key = prepared_profile_cache_key(pwm, batch)
        with open(tmp_path / key / "data.bin", "rb") as f:
            assert f.read(19) == b"MIMOSA-PREP-MMAP-1\0"
        with open(tmp_path / key / "meta.toml", "rb") as f:
            metadata = tomllib.load(f)
        assert metadata["format"] == "prepared_profile_mmap"
        assert metadata["n_rows"] == len(prepared.bundle.forward)

    def test_prepared_cache_streams_sections_without_legacy_payload_encoder(
        self, pwm, batch, tmp_path, monkeypatch
    ):
        from mimosa import cache as cache_module

        def fail(*args, **kwargs):
            raise AssertionError("prepared cache writes must stream sections")

        monkeypatch.setattr(cache_module, "_encode_prepared_profile_with_metadata", fail)
        cache = Cache(str(tmp_path))
        prepared = prepare_profile(pwm, batch, cache=cache)

        assert prepared == prepare_profile(pwm, batch, cache=Cache(str(tmp_path)))

    def test_prepared_cache_exposes_write_and_hit_phase_timings(
        self, pwm, batch, tmp_path
    ):
        timings = {}
        cache = Cache(str(tmp_path), timings=timings)
        prepare_profile(pwm, batch, cache=cache)
        assert {"cache_encode", "cache_lock_wait", "cache_write"} <= timings.keys()

        hit_timings = {}
        prepare_profile(pwm, batch, cache=Cache(str(tmp_path), timings=hit_timings))
        assert {"cache_checksum", "cache_semantic_validation"} <= hit_timings.keys()

    def test_prepare_miss_on_model_change(self, pwm, batch, tmp_path):
        from mimosa import PWM

        cache = Cache(str(tmp_path))
        p1 = prepare_profile(pwm, batch, cache=cache)
        w = pwm.weights.copy()
        w[0, 0] += 1.0
        pwm2 = PWM(pwm.name, w, pwm.background)
        p2 = prepare_profile(pwm2, batch, cache=cache)
        assert not np.array_equal(p1.bundle.forward.data, p2.bundle.forward.data)
