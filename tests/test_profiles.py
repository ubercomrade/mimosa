import numpy as np
import pytest

from mimosa import (
    BaMM,
    ComparisonResult,
    EncodedSequences,
    ProfileConfig,
    compare,
    compare_many,
    prepare_profile,
)
from mimosa.profiles.normalization import (
    EmpiricalLogTail,
    HybridEmpiricalLogTail,
    fit,
    normalization_fingerprint,
    transform_scores,
)
from mimosa.profiles.prepared import ScoreProfile
from mimosa.cache import Cache
from mimosa.io.fasta import read_fasta, read_scores
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm


def _lookup_score(table, score):
    from mimosa.profiles.normalization import LogTailTable, HybridLogTailTable

    if isinstance(table, LogTailTable):
        idx = min(
            max(
                0,
                int(np.searchsorted(-table.scores, -score, side="right")) - 1,
            ),
            table.scores.size - 1,
        )
        return table.log_tail[idx]
    if isinstance(table, HybridLogTailTable):
        if table.exact_tail.scores.size and score >= table.exact_tail.scores[-1]:
            return _lookup_score(table.exact_tail, score)
        if table.log_tail.size == 0:
            return np.float32(0.0)
        index = 0 if table.bin_width == 0 else int((float(score) - float(table.minimum)) / table.bin_width)
        index = min(max(index, 0), table.log_tail.size - 1)
        return table.log_tail[index]
    raise ValueError(f"unknown table type: {type(table)!r}")


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
        assert _lookup_score(table, np.float32(2.5)) == table.log_tail[0]
        assert _lookup_score(table, np.float32(0.0)) == table.log_tail[2]
        assert _lookup_score(table, np.float32(3.0)) == table.log_tail[0]
        assert _lookup_score(table, np.float32(4.0)) == table.log_tail[0]

    def test_empirical_uses_upper_tail_from_separate_calibration(self):
        from mimosa import RaggedArray, StrandPair
        from mimosa.profiles.normalization import _fit_normalize

        foreground = RaggedArray.from_rows([[2.5]])
        calibration = RaggedArray.from_rows([[1.0, 2.0, 3.0]])
        _, normalized = _fit_normalize(
            EmpiricalLogTail(),
            StrandPair(foreground, foreground),
            calibration=StrandPair(calibration, calibration),
        )

        assert normalized.forward[0][0] == pytest.approx(-np.log10(1 / 3))

    def test_hybrid_exact_tail_uses_upper_tail_lookup(self):
        from mimosa import RaggedArray
        from mimosa.profiles.normalization import HybridLogTailTable, LogTailTable

        exact = LogTailTable(
            np.array([3.0, 2.0, 1.0], dtype=np.float32),
            np.array([-np.log10(1 / 3), -np.log10(2 / 3), 0.0], dtype=np.float32),
        )
        table = HybridLogTailTable(
            0.0, 1.0, np.zeros(4, dtype=np.float32), exact
        )

        normalized = transform_scores(table, RaggedArray.from_rows([[2.5]]))
        assert normalized[0][0] == pytest.approx(-np.log10(1 / 3))

    def test_empty_scores(self):
        table = fit(EmpiricalLogTail(), np.array([], dtype=np.float32))
        assert table.scores.size == 1

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError):
            fit(EmpiricalLogTail(), np.array([1.0, np.nan], dtype=np.float32))

    def test_hybrid_constant_scores(self):
        table = fit(HybridEmpiricalLogTail(), np.full(100, 5.0, dtype=np.float32))
        assert table.bin_width == 1.0
        assert _lookup_score(table, np.float32(5.0)) == 0.0

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
                out[i], [_lookup_score(table, s) for s in scores[i]], rtol=1e-6
            )

    def test_hybrid_transform_matches_lookup(self):
        from mimosa import RaggedArray

        rng = np.random.default_rng(3)
        values = rng.normal(size=2000).astype(np.float32)
        scores = RaggedArray.from_rows([values[:1000], values[1000:]])
        table = fit(HybridEmpiricalLogTail(256), scores.data, tail_logerr=1.0)
        expected = np.array(
            [_lookup_score(table, value) for value in scores.data], dtype=np.float32
        )

        serial = transform_scores(table, scores)
        np.testing.assert_array_equal(serial.data, expected)

    def test_sparse_anchor_positions_do_not_allocate_full_candidate_buffer(
        self, monkeypatch
    ):
        from mimosa import RaggedArray
        from mimosa.profiles import anchors as anchor_module

        allocations = []
        original_empty = np.empty

        def tracked_empty(shape, *args, **kwargs):
            allocations.append(int(np.prod(shape)))
            return original_empty(shape, *args, **kwargs)

        scores = RaggedArray.from_rows(
            [np.concatenate(([1.0], np.zeros(9_999, dtype=np.float32)))]
        )
        monkeypatch.setattr(anchor_module.np, "empty", tracked_empty)
        anchors = anchor_module.collect_anchor_csr(scores, threshold=0.5)

        assert anchors.positions.nbytes == np.dtype(np.int32).itemsize
        assert anchors.positions.base is None
        assert 10_000 not in allocations

    def test_anchor_positions_fall_back_to_int64_for_large_coordinates(self):
        from mimosa import RaggedArray
        from mimosa.profiles.anchors import collect_anchor_csr

        scores = RaggedArray.from_rows([[1.0]])
        offset = int(np.iinfo(np.int32).max) + 1
        anchors = collect_anchor_csr(
            scores, threshold=0.5, position_offset=offset
        )

        assert anchors.positions.dtype == np.int64
        assert anchors.positions.tolist() == [offset]

    def test_fingerprint(self):
        assert normalization_fingerprint(EmpiricalLogTail()) == "empirical-log-tail-v1"
        assert (
            normalization_fingerprint(HybridEmpiricalLogTail(1024))
            == "hybrid-log-tail-v3;bins=1024"
        )

    def test_prepared_normalized_scores_are_threshold_independent(
        self, pwm_pair, batch
    ):
        lower = prepare_profile(pwm_pair[0], batch, min_logerr=1.0)
        higher = prepare_profile(pwm_pair[0], batch, min_logerr=2.0)

        np.testing.assert_array_equal(
            lower.bundle.forward.data, higher.bundle.forward.data
        )
        np.testing.assert_array_equal(
            lower.bundle.reverse.data, higher.bundle.reverse.data
        )
        assert lower.anchors[0].positions.size > higher.anchors[0].positions.size

    def test_prepared_exact_scores_use_separate_background(self, pwm_pair, batch):
        from mimosa import scan
        from mimosa.profiles.normalization import _fit_normalize

        background = read_fasta("examples/background.fa")[0]
        raw = scan(pwm_pair[0], batch, strands="both")
        calibration = scan(pwm_pair[0], background, strands="both")
        _, expected = _fit_normalize(
            EmpiricalLogTail(), raw, calibration=calibration
        )

        actual = prepare_profile(
            pwm_pair[0], batch, background=background, min_logerr=2.0
        )

        np.testing.assert_allclose(
            actual.bundle.forward.data, expected.forward.data, rtol=1e-6
        )
        np.testing.assert_allclose(
            actual.bundle.reverse.data, expected.reverse.data, rtol=1e-6
        )


class TestCompare:
    def test_prepared_profile_arrays_are_read_only(self, pwm_pair, batch):
        prepared = prepare_profile(pwm_pair[0], batch, min_logerr=1.0)
        arrays = (
            prepared.bundle.forward.data,
            prepared.bundle.forward.offsets,
            prepared.bundle.reverse.data,
            prepared.bundle.reverse.offsets,
            prepared.anchors[0].positions,
            prepared.anchors[0].offsets,
            prepared.anchors[1].positions,
            prepared.anchors[1].offsets,
        )
        assert all(not values.flags.writeable for values in arrays)
        with pytest.raises(ValueError):
            prepared.bundle.forward.data[0] = 0.0

    def test_equivalent_pwm_and_order_one_bamm_align_at_physical_zero(self):
        rng = np.random.default_rng(8)
        pwm_weights = rng.normal(size=(5, 4)).astype(np.float32)
        bamm_weights = np.empty((25, 4), dtype=np.float32)
        for code in range(25):
            bamm_weights[code] = pwm_weights[code % 5]
        from mimosa import PWM

        pwm = PWM("pwm", pwm_weights, (0.25, 0.25, 0.25, 0.25))
        bamm = BaMM("bamm", bamm_weights, 1, 4)
        sequences = EncodedSequences.from_rows(
            [rng.integers(0, 4, size=200, dtype=np.uint8) for _ in range(5)]
        )

        prepared_pwm = prepare_profile(pwm, sequences)
        prepared_bamm = prepare_profile(bamm, sequences)
        assert prepared_pwm.site_start_offset == 0
        assert prepared_bamm.site_start_offset == 1
        assert all(
            positions.size == 0 or positions.min() >= 1
            for positions in (
                prepared_bamm.anchors[0].positions,
                prepared_bamm.anchors[1].positions,
            )
        )

        for query, target in ((pwm, bamm), (bamm, pwm)):
            result = compare(
                query,
                target,
                sequences,
                search_range=0,
                window_radius=0,
            )
            assert result.offset == 0
            assert result.score == pytest.approx(1.0)

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
        compare_module._prepare_and_compare_with_threads(
            query, target, batch, None, query.min_logerr, query.normalization,
            ProfileConfig(), None, None, 1,
        )
        assert calls[0] == 1
        assert calls[1] >= 1

    def test_compare_many_applies_thread_budget_to_query_and_serial_target(
        self, pwm_pair, batch, monkeypatch
    ):
        import importlib
        import numba

        compare_module = importlib.import_module("mimosa.compare")
        observed_threads = []
        original = compare_module._prepare_profile

        def observed(*args, **kwargs):
            observed_threads.append(numba.get_num_threads())
            return original(*args, **kwargs)

        previous_threads = numba.get_num_threads()
        monkeypatch.setattr(compare_module, "_prepare_profile", observed)
        compare_many(
            pwm_pair[0],
            [pwm_pair[1]],
            batch,
            total_threads=2,
            inner_threads=2,
        )

        assert observed_threads == [2, 2]
        assert numba.get_num_threads() == previous_threads

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
                pwm_pair[0],
                [CustomModel(), CustomModel()],
                batch,
                total_threads=2,
                inner_threads=1,
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
        monkeypatch.setattr(
            alignment_module, "use_alignment_parallel", lambda *args, **kwargs: False
        )
        serial = compare(query, target, metric=metric)
        monkeypatch.setattr(
            alignment_module, "use_alignment_parallel", lambda *args, **kwargs: True
        )
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

    def test_prepared_normalization_mismatch_is_rejected(self, pwm_pair, batch):
        query = prepare_profile(pwm_pair[0], batch)
        target = prepare_profile(pwm_pair[1], batch)

        with pytest.raises(ValueError, match="requested normalization"):
            compare(query, target, normalization=EmpiricalLogTail())

    def test_result_dict_includes_zero_contributing_sites(self):
        result = ComparisonResult(
            "query", "target", np.float32(0.0), 0, "++", "co", 0
        )
        assert result.to_dict()["n_sites"] == 0

    @pytest.mark.parametrize("metric", ("co", "dice", "cosine"))
    def test_zero_norm_profiles_have_no_contributing_sites(self, metric):
        query = ScoreProfile("query", [np.ones(25, dtype=np.float32)])
        target = ScoreProfile("target", [np.ones(25, dtype=np.float32)])

        result = compare(query, target, metric=metric, window_radius=0)
        assert result.score == 0.0
        assert result.n_sites == 0

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
