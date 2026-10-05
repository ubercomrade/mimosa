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
from mimosa.io.bundles import content_fingerprint_float64
from mimosa.io.fasta import read_fasta
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm
from mimosa.profiles.normalization import HybridEmpiricalLogTail, normalization_fingerprint


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


TAU0 = 2.0


def make_dist(scores):
    raw_scores = np.array(scores, dtype=np.float64)
    return NullDistribution(
        strategy="profile",
        metric="jaccard",
        raw_scores=raw_scores,
        pairs=[("a", "b", float(s)) for s in scores],
        n_null=len(scores),
        n_models=2,
        shuffle=True,
        seed=127,
        sampling_version="original-shuffled-ordered-pairs-v3",
        model_collection_fingerprint="f" * 64,
        sequence_fingerprint="s" * 64,
        background_fingerprint="none",
        contract={
            "metric": "jaccard",
            "search_range": 5,
            "min_logerr": np.float32(4.0),
            "normalization_version": normalization_fingerprint(HybridEmpiricalLogTail()),
            "alignment_version": "circular-jaccard-v1",
            "sequence_fingerprint": "s" * 64,
            "background_fingerprint": "none",
            "raw_scores_fingerprint": content_fingerprint_float64(raw_scores),
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
                "profile", "jaccard", np.array([1.0]), [], 1, 2, True, 0, "v", None, "s", "n", {}
            )
        with pytest.raises(ValueError, match="must be positive"):
            NullDistribution(
                "profile", "jaccard", np.array([]), [], 0, 2, True, 0, "v", None, "s", "n", {}
            )

    def test_owns_and_freezes_raw_scores_contract_and_pairs(self):
        source = np.array([0.2, 0.4], dtype=np.float64)
        dist = make_dist(source)
        source[0] = 0.9

        np.testing.assert_array_equal(dist.raw_scores, [0.2, 0.4])
        assert not dist.raw_scores.flags.writeable
        assert isinstance(dist.pairs, tuple)
        with pytest.raises(ValueError):
            dist.raw_scores[0] = 0.9
        with pytest.raises(TypeError):
            dist.contract["metric"] = "other"

    def test_rejects_raw_scores_with_stale_fingerprint(self):
        dist = make_dist([0.2, 0.4])
        with pytest.raises(ValueError, match="fingerprint"):
            NullDistribution(
                **{
                    **dist.__dict__,
                    "raw_scores": np.array([0.3, 0.4]),
                    "pairs": (("a", "b", 0.3), ("a", "b", 0.4)),
                    "contract": dict(dist.contract),
                }
            )

    def test_build_null_reproducible(self, pwm_models, batch):
        d1 = build_null(pwm_models, sequences=batch, n_samples=30, seed=42, min_logerr=TAU0)
        d2 = build_null(pwm_models, sequences=batch, n_samples=30, seed=42, min_logerr=TAU0)
        np.testing.assert_array_equal(d1.raw_scores, d2.raw_scores)
        assert d1.pairs == d2.pairs

    def test_build_null_sampling_contract(self, pwm_models, batch):
        d = build_null(pwm_models, sequences=batch, n_samples=30, seed=42, min_logerr=TAU0)
        assert d.n_null == 30
        assert d.metric == "jaccard"
        assert d.contract["alignment_version"] == "circular-jaccard-v1"
        assert d.contract["search_range"] == 5
        assert d.contract["min_logerr"] == np.float32(TAU0)
        assert set(d.contract) == {
            "metric",
            "search_range",
            "min_logerr",
            "normalization_version",
            "alignment_version",
            "sequence_fingerprint",
            "background_fingerprint",
            "raw_scores_fingerprint",
        }
        assert d.sampling_version == "original-shuffled-ordered-pairs-v3"
        assert np.all(np.isfinite(d.raw_scores))
        assert all(
            pair[0].startswith(("original:", "shuffled:"))
            and pair[1].startswith(("original:", "shuffled:"))
            for pair in d.pairs
        )
        # sampled pairs obey the exclusion contract (checked at the item level)
        from mimosa.statistics import _next_null_work_item

        rng = np.random.default_rng(42)
        for _ in range(200):
            q, t = _next_null_work_item(3, rng)
            assert not (q < 3 and t < 3)
            assert not (q >= 3 and q == t)

    def test_rejects_removed_metric(self):
        dist = make_dist([0.2, 0.4])
        with pytest.raises(ValueError, match="jaccard"):
            NullDistribution(
                **{**dist.__dict__, "metric": "co", "contract": dict(dist.contract)}
            )

    def test_build_null_uses_circular_profiles(self, pwm_models, batch, monkeypatch):
        import importlib

        statistics_module = importlib.import_module("mimosa.statistics")
        original = statistics_module._prepare_profile
        flags = []

        def observed(*args, **kwargs):
            flags.append(kwargs["circular"])
            return original(*args, **kwargs)

        monkeypatch.setattr(statistics_module, "_prepare_profile", observed)
        build_null(pwm_models, sequences=batch, n_samples=3, seed=1, min_logerr=TAU0)
        assert flags == [True] * (2 * len(pwm_models))

    @pytest.mark.parametrize("min_logerr", (0.0, -1.0, np.nan))
    def test_build_null_rejects_non_positive_threshold(
        self, pwm_models, batch, min_logerr
    ):
        with pytest.raises(ValueError, match="min_logerr"):
            build_null(pwm_models, sequences=batch, min_logerr=min_logerr)

    def test_build_null_requires_pwm(self, batch):
        from mimosa import BaMM

        m = BaMM("b", np.zeros((5, 4), dtype=np.float32), 0, 4)
        with pytest.raises(ValueError):
            build_null([m, m], sequences=batch)

    def test_build_null_requires_two(self, pwm_models, batch):
        with pytest.raises(ValueError):
            build_null(pwm_models[:1], sequences=batch)

    def test_build_null_requires_positive_samples(self, pwm_models, batch):
        with pytest.raises(ValueError, match="must be positive"):
            build_null(pwm_models[:2], sequences=batch, n_samples=0)

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
    def test_annotate_sorts_the_null_scores_once(self, monkeypatch):
        dist = make_dist([0.5, 0.6, 0.7])
        calls = 0
        original = np.sort

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(np, "sort", counted)
        annotate_results(
            [
                ComparisonResult("a", "b", np.float32(0.6), 0, "++", "jaccard", 1),
                ComparisonResult("a", "c", np.float32(0.7), 0, "++", "jaccard", 1),
            ],
            dist,
        )
        assert calls == 1

    def test_annotate_results(self):
        dist = make_dist([0.5, 0.6, 0.7, 0.8, 0.9])
        results = [
            ComparisonResult("a", "b", np.float32(0.85), 0, "++", "jaccard", 5),
            ComparisonResult("a", "c", np.float32(0.95), 1, "++", "jaccard", 3),
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
        results = [ComparisonResult("a", "b", np.float32(0.95), 0, "++", "jaccard", 5)]
        annotated = annotate_results(results, dist, effective_number_of_targets=10)
        assert annotated[0].e_value == pytest.approx(annotated[0].p_value * 10)

    def test_annotate_metric_mismatch(self):
        dist = make_dist([0.5])
        results = [ComparisonResult("a", "b", np.float32(0.9), 0, "++", "other", 5)]
        with pytest.raises(ValueError):
            annotate_results(results, dist)

    def test_annotate_dict_serialization(self):
        dist = make_dist([0.5, 0.6])
        results = [ComparisonResult("a", "b", np.float32(0.9), 0, "++", "jaccard", 5)]
        annotated = annotate_results(results, dist)[0]
        d = annotated.to_dict()
        assert d["annotation_schema_version"] == 1
        assert "p-value" in d
        assert "adj.p-value" in d
        assert "E-value" in d
        assert "null_id" in d


class TestAnnotationSafety:
    def test_rejects_result_with_different_parameters(self):
        dist = make_dist([0.5, 0.6])
        ok = ComparisonResult(
            "a", "b", np.float32(0.9), 0, "++", "jaccard", 5,
            search_range=5, min_logerr=np.float32(4.0),
        )
        assert annotate_results([ok], dist)[0].p_value is not None
        for kwargs in ({"search_range": 3, "min_logerr": 4.0}, {"search_range": 5, "min_logerr": 1.0}):
            bad = ComparisonResult("a", "b", np.float32(0.9), 0, "++", "jaccard", 5, **kwargs)
            with pytest.raises(ValueError, match="does not match"):
                annotate_results([bad], dist)

    def test_annotated_result_copy(self):
        import copy

        result = ComparisonResult("a", "b", np.float32(0.9), 0, "++", "jaccard", 5)
        annotated = annotate_results([result], make_dist([0.5]))[0]
        assert copy.copy(annotated).p_value == annotated.p_value
        assert copy.deepcopy(annotated).target == "b"

    def test_null_distribution_replace(self):
        import dataclasses

        dist = make_dist([0.5, 0.6])
        assert dataclasses.replace(dist, seed=43).seed == 43

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_evalue_rejects_non_finite_or_negative(self, bad):
        from mimosa.statistics import evalue

        with pytest.raises(ValueError):
            evalue(0.5, bad)
