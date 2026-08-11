import numpy as np
import pytest

from mimosa import (
    ComparisonResult,
    ProfileConfig,
    compare,
    compare_many,
    prepare_profile,
)
from mimosa.profiles.normalization import (
    EmpiricalLogTail,
    HybridEmpiricalLogTail,
    fit,
    lookup_score,
    normalization_fingerprint,
    transform_scores,
)
from mimosa.profiles.prepared import ScoreProfile
from mimosa.cache import Cache
from mimosa.io.fasta import read_fasta, read_scores
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm


@pytest.fixture
def pwm_pair():
    models = []
    for f in ["foxa2.meme", "gata2.meme"]:
        name, pfm = read_meme(f"examples/{f}")
        models.append(pwm_from_pfm(pfm, name=name))
    return models


@pytest.fixture
def batch():
    return read_fasta("examples/foreground.fa")[0]


class TestNormalization:
    def test_empirical_table(self):
        scores = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32)
        table = fit(EmpiricalLogTail(), scores)
        # unique desc: 3,2,1 with counts 3,2,1; cumulative tails 3,5,6
        assert table.scores.tolist() == [3.0, 2.0, 1.0]
        expected = [-np.log10(3 / 6), -np.log10(5 / 6), -np.log10(6 / 6)]
        np.testing.assert_allclose(table.log_tail, expected, rtol=1e-6)

    def test_empirical_lookup(self):
        scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        table = fit(EmpiricalLogTail(), scores)
        assert lookup_score(table, np.float32(2.5)) == table.log_tail[1]
        assert lookup_score(table, np.float32(0.0)) == table.log_tail[2]
        assert lookup_score(table, np.float32(3.0)) == table.log_tail[0]

    def test_empty_scores(self):
        table = fit(EmpiricalLogTail(), np.array([], dtype=np.float32))
        assert table.scores.size == 1

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError):
            fit(EmpiricalLogTail(), np.array([1.0, np.nan], dtype=np.float32))

    def test_hybrid_constant_scores(self):
        table = fit(HybridEmpiricalLogTail(), np.full(100, 5.0, dtype=np.float32))
        assert table.bin_width == 1.0
        assert lookup_score(table, np.float32(5.0)) == 0.0

    def test_hybrid_histogram_shape(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=1000).astype(np.float32)
        table = fit(HybridEmpiricalLogTail(256), scores)
        assert table.log_tail.size == 256

    def test_hybrid_exact_tail(self):
        rng = np.random.default_rng(2)
        scores = rng.normal(size=10000).astype(np.float32)
        table = fit(HybridEmpiricalLogTail(), scores, tail_logerr=1.0)
        # top ~10% should be exact
        assert table.exact_tail.scores.size > 500

    def test_transform_scores(self):
        from mimosa import RaggedArray

        scores = RaggedArray.from_rows([[1.0, 3.0], [2.0]])
        table = fit(EmpiricalLogTail(), scores.data)
        out = transform_scores(table, scores)
        for i in range(len(scores)):
            np.testing.assert_allclose(
                out[i], [lookup_score(table, s) for s in scores[i]], rtol=1e-6
            )

    def test_hybrid_transform_matches_lookup(self):
        from mimosa import RaggedArray

        rng = np.random.default_rng(3)
        values = rng.normal(size=2000).astype(np.float32)
        scores = RaggedArray.from_rows([values[:1000], values[1000:]])
        table = fit(HybridEmpiricalLogTail(256), scores.data, tail_logerr=1.0)
        expected = np.array(
            [lookup_score(table, value) for value in scores.data], dtype=np.float32
        )

        serial = transform_scores(table, scores)
        np.testing.assert_array_equal(serial.data, expected)

    def test_fingerprint(self):
        assert normalization_fingerprint(EmpiricalLogTail()) == "empirical-log-tail-v1"
        assert (
            normalization_fingerprint(HybridEmpiricalLogTail(1024))
            == "hybrid-log-tail-v2;bins=1024"
        )


class TestCompare:
    def test_compare_models_matches_reference(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        result = compare(m1, m2, batch)
        assert isinstance(result, ComparisonResult)
        assert result.query == "MA0047.3"
        assert result.target == "MA0036.2"
        assert result.orientation == "++"
        assert result.offset == -3
        assert result.n_sites == 199
        assert abs(float(result.score) - 0.71147388) < 1e-5

    def test_compare_score_profiles(self):
        s1 = read_scores("examples/scores_1.fasta")
        s2 = read_scores("examples/scores_2.fasta")
        result = compare(s1, s2)
        assert result.offset == -6
        assert result.orientation == "++"
        assert result.n_sites == 1719
        assert abs(float(result.score) - 0.55700350) < 1e-5
        dice = compare(s1, s2, metric="dice")
        assert abs(float(dice.score) - 0.47674134) < 1e-5

    def test_compare_self_is_perfect(self, pwm_pair, batch):
        m1, _ = pwm_pair
        result = compare(m1, m1, batch)
        assert float(result.score) == 1.0
        assert result.offset == 0
        assert result.orientation == "++"

    def test_prepared_reuse(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = prepare_profile(m1, batch)
        direct = compare(m1, m2, batch)
        prepared = compare(query, m2, batch)
        assert prepared == direct

    def test_one_to_many(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = prepare_profile(m1, batch)
        results = compare_many(query, [m2, m2], batch)
        assert len(results) == 2
        assert results[0] == results[1]

    def test_compare_many_preserves_target_order(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = prepare_profile(m1, batch)
        target = prepare_profile(m2, batch)
        targets = [target, query, target, query, target]
        expected = [compare(query, target) for target in targets]
        assert compare_many(query, targets, batch) == expected

    @pytest.mark.parametrize(
        "total_threads, inner_threads",
        ((1, 1), (4, 1), (4, 2), (4, 4)),
    )
    def test_compare_many_budgets_match_serial_and_preserve_order(
        self, pwm_pair, batch, total_threads, inner_threads
    ):
        query = prepare_profile(pwm_pair[0], batch)
        target = prepare_profile(pwm_pair[1], batch)
        targets = [target, query, target, query]
        expected = compare_many(query, targets, batch)
        assert compare_many(
            query,
            targets,
            batch,
            total_threads=total_threads,
            inner_threads=inner_threads,
        ) == expected

    def test_joblib_worker_disables_inner_numba_threads(self, pwm_pair, batch, monkeypatch):
        import importlib
        import numba

        compare_module = importlib.import_module("mimosa.compare")
        calls = []
        query = prepare_profile(pwm_pair[0], batch)
        target = prepare_profile(pwm_pair[1], batch)
        monkeypatch.setattr(numba, "set_num_threads", calls.append)
        compare_module._compare_prepared_with_threads(query, target, ProfileConfig(), 1)
        assert calls == [1]

    @pytest.mark.parametrize("threshold", (0.0, 1.0))
    @pytest.mark.parametrize("metric", ("co", "dice", "cosine"))
    def test_compare_many_joblib_matches_serial(
        self, pwm_pair, batch, tmp_path, metric, threshold
    ):
        query = prepare_profile(pwm_pair[0], batch, min_logerr=threshold)
        targets = [pwm_pair[1], pwm_pair[0], pwm_pair[1]]
        serial = compare_many(
            query,
            targets,
            batch,
            metric=metric,
            min_logerr=threshold,
        )
        parallel = compare_many(
            query,
            targets,
            batch,
            metric=metric,
            min_logerr=threshold,
            cache=Cache(str(tmp_path)),
            total_threads=2,
            inner_threads=1,
        )
        assert parallel == serial

    def test_compare_many_joblib_cold_and_disk_cache_match(self, pwm_pair, batch, tmp_path):
        targets = [pwm_pair[1], pwm_pair[1]]
        expected = compare_many(pwm_pair[0], targets, batch)
        cold = compare_many(
            pwm_pair[0], targets, batch, cache=Cache(str(tmp_path)), total_threads=2
        )
        disk = compare_many(
            pwm_pair[0], targets, batch, cache=Cache(str(tmp_path)), total_threads=2
        )
        assert cold == disk == expected

    @pytest.mark.parametrize("total_threads", (False, 0, -1, 1.5, "2"))
    def test_compare_many_rejects_invalid_total_threads(self, pwm_pair, batch, total_threads):
        with pytest.raises((TypeError, ValueError), match="total_threads"):
            compare_many(pwm_pair[0], [pwm_pair[1]], batch, total_threads=total_threads)

    @pytest.mark.parametrize("inner_threads", (False, 0, -1, 5, 1.5, "2"))
    def test_compare_many_rejects_invalid_inner_threads(self, pwm_pair, batch, inner_threads):
        with pytest.raises((TypeError, ValueError), match="inner_threads"):
            compare_many(pwm_pair[0], [pwm_pair[1]], batch, inner_threads=inner_threads)

    def test_compare_many_rejects_non_divisible_budget(self, pwm_pair, batch):
        with pytest.raises(ValueError, match="divisible"):
            compare_many(pwm_pair[0], [pwm_pair[1]], batch, total_threads=3, inner_threads=2)

    def test_compare_many_rejects_raw_custom_model_in_joblib_path(
        self, pwm_pair, batch
    ):
        from mimosa import MotifModel

        class CustomModel(MotifModel):
            name = "custom"
            motif_length = 1

            def scan_into(self, sequence, forward, reverse, /):
                forward.fill(0)
                reverse.fill(0)

        with pytest.raises(TypeError, match="custom models"):
            compare_many(
                pwm_pair[0], [CustomModel()], batch, total_threads=2, inner_threads=1
            )

    def test_compare_many_reprepares_duplicate_raw_targets_without_cache(
        self, pwm_pair, batch, monkeypatch
    ):
        import importlib

        compare_module = importlib.import_module("mimosa.compare")

        calls = 0
        original = compare_module._prepare_profile

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(compare_module, "_prepare_profile", counted)
        query = prepare_profile(pwm_pair[0], batch)
        compare_many(query, [pwm_pair[1], pwm_pair[1]], batch)
        assert calls == 2

    def test_compare_many_reuses_sequence_context(self, pwm_pair, batch, tmp_path, monkeypatch):
        from mimosa import cache as cache_module
        from mimosa.cache import Cache

        calls = 0
        original = cache_module.sequence_fingerprint

        def counted(value):
            nonlocal calls
            calls += 1
            return original(value)

        monkeypatch.setattr(cache_module, "sequence_fingerprint", counted)
        cache = Cache(str(tmp_path))
        compare_many(pwm_pair[0], [pwm_pair[1]] * 3, batch, cache=cache)
        assert calls == 1

    def test_compare_many_matches_direct(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = prepare_profile(m1, batch)
        target = prepare_profile(m2, batch)
        targets = [target] * 2
        for metric in ("co", "dice", "cosine"):
            results = compare_many(query, targets, batch, metric=metric)
            expected = compare(query, target, batch, metric=metric)
            assert results[0] == expected
            assert all(result == expected for result in results)

    def test_compare_many_prepared_positive_threshold_matches_serial(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = prepare_profile(m1, batch, min_logerr=1.0)
        target = prepare_profile(m2, batch, min_logerr=1.0)
        targets = [target] * 2
        serial = [compare(query, target, metric="co", min_logerr=1.0) for target in targets]
        results = compare_many(query, targets, batch, min_logerr=1.0)
        assert results == serial

    def test_compare_many_rejects_incompatible_prepared_target(self, pwm_pair, batch):
        query = prepare_profile(pwm_pair[0], batch, min_logerr=0.0)
        target = prepare_profile(pwm_pair[1], batch, min_logerr=1.0)
        with pytest.raises(ValueError, match="different min_logerr"):
            compare_many(query, [target])

    def test_compare_rejects_different_row_counts(self):
        query = ScoreProfile("query", [[1.0, 2.0], [3.0, 4.0]])
        target = ScoreProfile("target", [[1.0, 2.0]])
        with pytest.raises(ValueError, match="same number of rows"):
            compare(query, target)

    def test_compare_many_prepared_shared_strands(self, pwm_pair, batch):
        from mimosa.profiles.prepared import ScoreProfile

        m1, m2 = pwm_pair
        prepared_target = prepare_profile(m2, batch)
        shared_query = prepare_profile(
            ScoreProfile("shared-query", prepared_target.bundle.forward),
        )
        prepared_query = prepare_profile(m1, batch)
        shared_target = prepare_profile(
            ScoreProfile("shared-target", prepared_query.bundle.forward),
        )
        for query, targets in (
            (shared_query, [prepared_target] * 2),
            (prepared_query, [shared_target] * 2),
        ):
            serial = [compare(query, target, metric="cosine") for target in targets]
            results = compare_many(query, targets, batch, metric="cosine")
            assert results == serial

    @pytest.mark.parametrize("metric", ("co", "dice", "cosine"))
    @pytest.mark.parametrize("threshold", (0.0, 1.0))
    def test_alignment_row_parallel_matches_serial(
        self, pwm_pair, batch, monkeypatch, metric, threshold
    ):
        import importlib

        alignment_module = importlib.import_module("mimosa.profiles.alignment")
        query = prepare_profile(pwm_pair[0], batch, min_logerr=threshold)
        target = prepare_profile(pwm_pair[1], batch, min_logerr=threshold)
        monkeypatch.setattr(alignment_module, "use_parallel", lambda *args, **kwargs: False)
        serial = compare(query, target, metric=metric)
        monkeypatch.setattr(alignment_module, "use_parallel", lambda *args, **kwargs: True)
        parallel = compare(query, target, metric=metric)
        assert parallel == serial

    def test_prepared_profile_picklable(self, pwm_pair, batch):
        import pickle

        m1, _ = pwm_pair
        prepared = prepare_profile(m1, batch)
        roundtrip = pickle.loads(pickle.dumps(prepared))
        assert roundtrip == prepared

    def test_prepared_profile_rejects_negative_anchor(self, pwm_pair, batch):
        from mimosa.arrays import RaggedArray, StrandPair
        from mimosa.profiles.anchors import AnchorCSR
        from mimosa.profiles.prepared import PreparedProfile

        prepared = prepare_profile(pwm_pair[0], batch)
        bad = AnchorCSR(np.array([-1], dtype=np.int64), np.array([0, 1], dtype=np.int64))
        with pytest.raises(ValueError):
            PreparedProfile(
                prepared.name,
                StrandPair(RaggedArray.from_rows([[1.0]]), RaggedArray.from_rows([[1.0]])),
                (bad, bad),
            )

    def test_metric_variants(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        co = compare(m1, m2, batch, metric="co")
        dice = compare(m1, m2, batch, metric="dice")
        cos = compare(m1, m2, batch, metric="cosine")
        assert co.metric == "co"
        assert dice.metric == "dice"
        assert cos.metric == "cosine"
        assert 0 <= float(dice.score) <= 1

    def test_removed_rowwise_metric_names(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        for metric in ("co_rowwise", "dice_rowwise"):
            with pytest.raises(ValueError):
                compare(m1, m2, batch, metric=metric)

    def test_min_logerr_threshold_mode(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        r = compare(m1, m2, batch, min_logerr=2.0)
        assert r.n_sites > 0

    def test_prepared_threshold_mismatch(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = prepare_profile(m1, batch, min_logerr=1.0)
        with pytest.raises(ValueError):
            compare(query, m2, batch, min_logerr=2.0)

    def test_mixed_unsupported(self, pwm_pair, batch):
        m1, _ = pwm_pair
        s1 = read_scores("examples/scores_1.fasta")
        with pytest.raises(ValueError):
            compare(s1, m1, batch)


class TestTieBreaking:
    def test_config_defaults(self):
        config = ProfileConfig()
        assert config.search_range == 10
        assert config.window_radius == 10
        assert config.realign_window == 3
        assert config.min_logerr == 0.0

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            ProfileConfig(search_range=-1)
        with pytest.raises(ValueError):
            ProfileConfig(min_logerr=np.nan)


class TestScoreProfile:
    def test_name_and_rows(self):
        s = read_scores("examples/scores_1.fasta")
        assert s.name == "scores_1"
        assert len(s) == 1000

    def test_nonfinite_rejected(self):
        from mimosa import RaggedArray
        from mimosa.errors import ModelFormatError

        ra = RaggedArray.from_rows([[1.0, np.nan]])
        with pytest.raises(ModelFormatError):
            ScoreProfile("bad", ra)

    def test_flat_scores_are_not_rows(self):
        with pytest.raises(TypeError):
            ScoreProfile("flat", np.array([1.0, 2.0], dtype=np.float32))
