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
    def test_tie_ladder(self, pwm_pair, batch):
        m1, m2 = pwm_pair
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
