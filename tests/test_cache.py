import os
import multiprocessing
import pickle

import numpy as np
import pytest

from mimosa.cache import (
    Cache,
    cache_get,
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
    def test_set_get(self, tmp_path):
        cache = Cache(str(tmp_path))
        cache_set(cache, "abc123", b"payload")
        assert cache_get(cache, "abc123") == b"payload"

    def test_disabled(self, tmp_path):
        cache = Cache(str(tmp_path), enabled=False)
        cache_set(cache, "abc", b"data")
        assert cache_get(cache, "abc") is None

    def test_missing(self, tmp_path):
        cache = Cache(str(tmp_path))
        assert cache_get(cache, "nope") is None

    def test_corrupt_checksum(self, tmp_path):
        cache = Cache(str(tmp_path))
        cache_set(cache, "abc", b"payload")
        entry = os.path.join(str(tmp_path), "abc")
        data_path = os.path.join(entry, "data.bin")
        with open(data_path, "r+b") as f:
            f.seek(0)
            f.write(bytes([f.read(1)[0] ^ 0xFF]))
        assert cache_get(cache, "abc") is None

    def test_clear(self, tmp_path):
        cache = Cache(str(tmp_path))
        cache_set(cache, "a", b"1")
        cache_set(cache, "b", b"2")
        assert clearcache(cache) == 2
        assert cache_get(cache, "a") is None

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
        from mimosa.cache import Cache

        cache = Cache("/tmp/opencode/cache-test")
        k1 = prepared_profile_cache_key(cache, pwm, batch)
        k2 = prepared_profile_cache_key(cache, pwm, batch)
        assert k1 == k2
        assert len(k1) == 16

    def test_key_changes_with_model(self, pwm, batch):
        from mimosa.cache import Cache

        cache = Cache("/tmp/opencode/cache-test")
        k1 = prepared_profile_cache_key(cache, pwm, batch)
        from mimosa import PWM

        m2 = PWM("other", pwm.weights, pwm.background)
        k2 = prepared_profile_cache_key(cache, m2, batch)
        assert k1 != k2

    def test_key_changes_with_min_logerr(self, pwm, batch):
        from mimosa.cache import Cache

        cache = Cache("/tmp/opencode/cache-test")
        k1 = prepared_profile_cache_key(cache, pwm, batch, min_logerr=0.0)
        k2 = prepared_profile_cache_key(cache, pwm, batch, min_logerr=2.0)
        assert k1 != k2

    def test_prepare_hit(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        p1 = prepare_profile(pwm, batch, cache=cache)
        p2 = prepare_profile(pwm, batch, cache=cache)
        assert p1 == p2

    def test_prepare_payload_is_pickle(self, pwm, batch, tmp_path):
        cache = Cache(str(tmp_path))
        prepared = prepare_profile(pwm, batch, cache=cache)
        key = prepared_profile_cache_key(cache, pwm, batch)
        with open(tmp_path / key / "data.bin", "rb") as f:
            assert pickle.loads(f.read()) == prepared

    def test_prepare_miss_on_model_change(self, pwm, batch, tmp_path):
        from mimosa import PWM

        cache = Cache(str(tmp_path))
        p1 = prepare_profile(pwm, batch, cache=cache)
        w = pwm.weights.copy()
        w[0, 0] += 1.0
        pwm2 = PWM(pwm.name, w, pwm.background)
        p2 = prepare_profile(pwm2, batch, cache=cache)
        assert not np.array_equal(p1.bundle.forward.data, p2.bundle.forward.data)
