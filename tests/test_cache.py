import os

import numpy as np
import pytest

from mimosa.cache import Cache, cache_get, cache_set, clearcache, prepared_profile_cache_key
from mimosa.io.fasta import read_fasta
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm
from mimosa.profiles.prepared import prepare_profile


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

    def test_prepare_miss_on_model_change(self, pwm, batch, tmp_path):
        from mimosa import PWM

        cache = Cache(str(tmp_path))
        p1 = prepare_profile(pwm, batch, cache=cache)
        w = pwm.weights.copy()
        w[0, 0] += 1.0
        pwm2 = PWM(pwm.name, w, pwm.background)
        p2 = prepare_profile(pwm2, batch, cache=cache)
        assert not np.array_equal(p1.bundle.forward.data, p2.bundle.forward.data)
