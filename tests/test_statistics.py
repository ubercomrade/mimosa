import numpy as np
import pytest

from mimosa import (
    AnnotatedResult,
    NullDistribution,
    adjusted_pvalues,
    annotate_results,
    build_null,
    empirical_upper_tail_pvalue,
    evalue,
)
from mimosa.compare import ComparisonResult
from mimosa.io.fasta import read_fasta
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm


@pytest.fixture
def pwm_models():
    models = []
    for f in ["foxa2.meme", "gata2.meme", "gata4.meme"]:
        name, pfm = read_meme(f"examples/{f}")
        models.append(pwm_from_pfm(pfm, name=name))
    return models


@pytest.fixture
def batch():
    return read_fasta("examples/foreground.fa")[0]


def make_dist(scores):
    return NullDistribution(
        strategy="profile",
        metric="co",
        raw_scores=np.array(scores, dtype=np.float64),
        pairs=[("a", "b", float(s)) for s in scores],
        n_null=len(scores),
        n_models=2,
        model_type="pwm",
        shuffle=True,
        seed=127,
        sampling_version="original-shuffled-ordered-pairs-v3",
        model_collection_fingerprint="f" * 64,
        sequence_fingerprint="s" * 64,
        background_fingerprint="none",
        contract={
            "metric": "co",
            "search_range": 10,
            "window_radius": 10,
            "realign_window": 3,
            "min_logerr": np.float32(0.0),
            "normalization_version": "hybrid-log-tail-v2;bins=65536",
            "alignment_version": "profile-alignment-v1",
            "sequence_fingerprint": "s" * 64,
            "background_fingerprint": "none",
            "raw_scores_fingerprint": "r" * 64,
        },
    )


class TestPValues:
    def test_empirical_upper_tail(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        # score 2.5: 2 scores >= 2.5 -> (2+1)/(4+1) = 0.6
        assert empirical_upper_tail_pvalue(scores, 2.5) == pytest.approx(0.6)
        assert empirical_upper_tail_pvalue(scores, 4.0) == pytest.approx(0.4)
        assert empirical_upper_tail_pvalue(scores, 0.0) == pytest.approx(1.0)
        assert empirical_upper_tail_pvalue(scores, 5.0) == pytest.approx(0.2)

    def test_empirical_empty(self):
        with pytest.raises(ValueError):
            empirical_upper_tail_pvalue(np.array([]), 1.0)

    def test_evalue(self):
        assert evalue(0.01, 100) == 1.0
        assert evalue(0.5, 0) == 0.0
        with pytest.raises(ValueError):
            evalue(1.5, 10)
        with pytest.raises(ValueError):
            evalue(0.5, -1)

    def test_adjusted_pvalues_bh(self):
        p = np.array([0.01, 0.04, 0.2])
        adj = adjusted_pvalues(p)
        # BH: sorted 0.01,0.04,0.2; q3=0.2, q2=min(0.2,0.04*3/2)=0.06, q1=min(0.06,0.03)=0.03
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.2], atol=1e-12)

    def test_adjusted_empty(self):
        assert adjusted_pvalues(np.array([])).size == 0

    def test_adjusted_invalid(self):
        with pytest.raises(ValueError):
            adjusted_pvalues(np.array([1.5]))


class TestNullDistribution:
    def test_validation(self):
        with pytest.raises(ValueError):
            NullDistribution(
                "profile", "co", np.array([1.0]), [], 1, 2, "pwm", True, 0, "v", None, "s", "n", {}
            )

    def test_build_null_reproducible(self, pwm_models, batch):
        d1 = build_null(pwm_models, sequences=batch, n_samples=30, seed=42)
        d2 = build_null(pwm_models, sequences=batch, n_samples=30, seed=42)
        np.testing.assert_array_equal(d1.raw_scores, d2.raw_scores)
        assert d1.pairs == d2.pairs

    def test_build_null_sampling_contract(self, pwm_models, batch):
        d = build_null(pwm_models, sequences=batch, n_samples=30, seed=42)
        assert d.n_null == 30
        assert d.metric == "co"
        assert d.model_type == "pwm"
        assert d.sampling_version == "original-shuffled-ordered-pairs-v3"
        assert np.all(np.isfinite(d.raw_scores))
        # sampled pairs obey the exclusion contract (checked at the item level)
        from mimosa.statistics import _next_null_work_item

        rng = np.random.default_rng(42)
        for _ in range(200):
            q, t = _next_null_work_item(3, rng)
            assert not (q < 3 and t < 3)
            assert not (q >= 3 and q == t)

    def test_build_null_requires_pwm(self, batch):
        from mimosa import BaMM

        m = BaMM("b", np.zeros((5, 4), dtype=np.float32), 0, 4)
        with pytest.raises(ValueError):
            build_null([m, m], sequences=batch)

    def test_build_null_requires_two(self, pwm_models, batch):
        with pytest.raises(ValueError):
            build_null(pwm_models[:1], sequences=batch)

    def test_build_null_duplicate_names(self, pwm_models, batch):
        from mimosa import PWM

        m = PWM("dup", pwm_models[0].weights, pwm_models[0].background)
        with pytest.raises(ValueError):
            build_null([m, m], sequences=batch)

    def test_shuffle_preserves_background(self, pwm_models):
        from mimosa.statistics import _shuffle_null_model

        m = pwm_models[0]
        shuffled = _shuffle_null_model(m, 12345)
        assert shuffled.background == m.background
        assert shuffled.motif_length == m.motif_length
        # N row is min of concrete rows
        np.testing.assert_allclose(
            shuffled.weights[4], shuffled.weights[:4].min(axis=0), atol=1e-6
        )


class TestAnnotation:
    def test_annotate_results(self):
        dist = make_dist([0.5, 0.6, 0.7, 0.8, 0.9])
        results = [
            ComparisonResult("a", "b", np.float32(0.85), 0, "++", "co", 5),
            ComparisonResult("a", "c", np.float32(0.95), 1, "++", "co", 3),
        ]
        annotated = annotate_results(results, dist)
        assert len(annotated) == 2
        assert all(isinstance(a, AnnotatedResult) for a in annotated)
        # null = [0.5..0.9]; score 0.95: 0 scores >= -> (0+1)/6 = 1/6
        # score 0.85: 1 score >= (0.9) -> (1+1)/6 = 1/3
        assert annotated[1].p_value == pytest.approx(1 / 6)
        assert annotated[0].p_value == pytest.approx(1 / 3)
        assert annotated[1].null_n == 5
        assert annotated[1].null_estimator == "empirical_upper_tail"
        assert annotated[1].null_id == annotated[0].null_id

    def test_annotate_effective_targets(self):
        dist = make_dist([0.5, 0.6, 0.7, 0.8, 0.9])
        results = [ComparisonResult("a", "b", np.float32(0.95), 0, "++", "co", 5)]
        annotated = annotate_results(results, dist, effective_number_of_targets=10)
        assert annotated[0].e_value == pytest.approx(annotated[0].p_value * 10)

    def test_annotate_metric_mismatch(self):
        dist = make_dist([0.5])
        results = [ComparisonResult("a", "b", np.float32(0.9), 0, "++", "dice", 5)]
        with pytest.raises(ValueError):
            annotate_results(results, dist)

    def test_annotate_dict_serialization(self):
        dist = make_dist([0.5, 0.6])
        results = [ComparisonResult("a", "b", np.float32(0.9), 0, "++", "co", 5)]
        annotated = annotate_results(results, dist)[0]
        d = annotated.to_dict()
        assert d["annotation_schema_version"] == 1
        assert "p-value" in d
        assert "adj.p-value" in d
        assert "E-value" in d
        assert "null_id" in d
