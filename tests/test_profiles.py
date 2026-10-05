import numpy as np
import pytest

from mimosa import (
    BaMM,
    ComparisonResult,
    EncodedSequences,
    PWM,
    PreparedProfile,
    ProfileConfig,
    RaggedArray,
    StrandPair,
    compare,
    compare_many,
    prepare_profile,
)
from mimosa.profiles.anchors import collect_anchor_csr, collect_both_anchors
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


# The example FASTA holds about 1.5e5 positions per strand, so tests use a
# threshold that keeps well over the recommended number of positions.
TAU0 = 2.0
EXAMPLE_U = [0.2, 4.8, 1.1, 0.4, 5.3, 0.0, 4.1, 0.7, 2.2, 0.3]
EXAMPLE_V = [0.5, 4.4, 0.2, 0.1, 3.6, 0.9, 0.3, 0.6, 0.1, 4.5]


def circular_profile(model, sequences, min_logerr=TAU0, **kwargs):
    return prepare_profile(
        model, sequences, min_logerr=min_logerr, circular=True, **kwargs
    )


def ready_profile(name, forward, reverse=None, tau0=TAU0):
    """Circular PreparedProfile from logERR rows, bypassing normalization."""
    forward = RaggedArray.from_rows(forward)
    reverse = forward if reverse is None else RaggedArray.from_rows(reverse)
    bundle = StrandPair(forward, reverse)
    return PreparedProfile(
        name, bundle, collect_both_anchors(bundle, tau0), tau0, circular=True
    )


def dense_jaccard(u_rows, v_rows, shift, tau0):
    """All-position reference: pair u[i] with v[(i + shift) mod L] per row."""
    min_sum = max_sum = 0.0
    n_sites = 0
    for u, v in zip(u_rows, v_rows):
        u = np.asarray(u, dtype=np.float32).astype(np.float64)
        v = np.roll(np.asarray(v, dtype=np.float32), -shift).astype(np.float64)
        keep = np.maximum(u, v) >= np.float32(tau0)
        min_sum += np.minimum(u, v)[keep].sum()
        max_sum += np.maximum(u, v)[keep].sum()
        n_sites += int(keep.sum())
    return (min_sum / max_sum if max_sum > 0 else 0.0), n_sites


def random_logerr_rows(rng, lengths):
    return [rng.exponential(1.2, size=n).astype(np.float32) for n in lengths]


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


# These hand-built profiles are far smaller than a real comparison.
@pytest.mark.filterwarnings("ignore:only about")
class TestJaccard:
    @pytest.mark.parametrize(
        "tau0, expected, n_sites",
        ((4.0, 8.6 / 18.7, 4), (2.0, 8.7 / 20.9, 5)),
    )
    def test_worked_example(self, tau0, expected, n_sites):
        result = compare(
            ready_profile("u", [EXAMPLE_U], tau0=tau0),
            ready_profile("v", [EXAMPLE_V], tau0=tau0),
            search_range=0,
        )
        assert float(result.score) == pytest.approx(expected, abs=1e-6)
        assert round(float(result.score), 4) == (0.4599 if tau0 == 4.0 else 0.4163)
        assert result.n_sites == n_sites
        assert (result.offset, result.orientation) == (0, "++")
        assert result.metric == "jaccard"

    def test_shift_is_circular(self):
        u = np.array(EXAMPLE_U, dtype=np.float32)
        result = compare(
            ready_profile("u", [u]), ready_profile("v", [np.roll(u, 3)]), search_range=4
        )
        assert float(result.score) == 1.0
        assert result.offset == 3

    @pytest.mark.parametrize("tau0", (0.5, 2.0, 4.0))
    def test_matches_dense_reference_on_ragged_rows(self, tau0):
        rng = np.random.default_rng(11)
        lengths = [40, 1, 0, 7, 120, 3]
        search_range = 5
        q = [random_logerr_rows(rng, lengths) for _ in range(2)]
        t = [random_logerr_rows(rng, lengths) for _ in range(2)]
        result = compare(
            ready_profile("q", q[0], q[1], tau0=tau0),
            ready_profile("t", t[0], t[1], tau0=tau0),
            search_range=search_range,
        )

        dense = {
            (label, shift): dense_jaccard(q[a], t[b], shift, tau0)
            for label, a, b in (("++", 0, 0), ("+-", 0, 1), ("-+", 1, 0), ("--", 1, 1))
            for shift in range(-search_range, search_range + 1)
        }
        best = max(score for score, _ in dense.values())
        score, n_sites = dense[(result.orientation, result.offset)]
        assert score == pytest.approx(best, abs=1e-12)
        assert float(result.score) == pytest.approx(score, abs=1e-6)
        assert result.n_sites == n_sites

    def test_kernel_matches_dense_reference_for_every_shift(self):
        from mimosa._kernels import _jaccard_orientation

        rng = np.random.default_rng(12)
        lengths = [30, 2, 0, 9, 64]
        search_range = 12
        q = ready_profile("q", random_logerr_rows(rng, lengths))
        t = ready_profile("t", random_logerr_rows(rng, lengths))
        scores = np.empty(2 * search_range + 1, dtype=np.float64)
        sites = np.empty(2 * search_range + 1, dtype=np.int64)
        _jaccard_orientation(
            q.bundle.forward.data, t.bundle.forward.data, q.bundle.forward.offsets,
            q.anchors[0].positions, q.anchors[0].offsets,
            t.anchors[0].positions, t.anchors[0].offsets,
            search_range, np.float32(TAU0), scores, sites,
        )
        q_rows = [q.bundle.forward[row] for row in range(len(lengths))]
        t_rows = [t.bundle.forward[row] for row in range(len(lengths))]
        for index, shift in enumerate(range(-search_range, search_range + 1)):
            score, n_sites = dense_jaccard(q_rows, t_rows, shift, TAU0)
            assert scores[index] == pytest.approx(score, abs=1e-12)
            assert sites[index] == n_sites

    def test_serial_and_parallel_kernels_are_bitwise_identical(self):
        from mimosa._kernels import _jaccard_orientation, _jaccard_orientation_parallel

        rng = np.random.default_rng(13)
        lengths = rng.integers(0, 90, size=300).tolist()
        search_range = 5
        n_shifts = 2 * search_range + 1
        q = ready_profile("q", random_logerr_rows(rng, lengths), tau0=1.0)
        t = ready_profile("t", random_logerr_rows(rng, lengths), tau0=1.0)
        args = (
            q.bundle.forward.data, t.bundle.forward.data, q.bundle.forward.offsets,
            q.anchors[0].positions, q.anchors[0].offsets,
            t.anchors[0].positions, t.anchors[0].offsets,
            search_range, np.float32(1.0),
        )
        serial = (np.empty(n_shifts), np.empty(n_shifts, dtype=np.int64))
        parallel = (np.empty(n_shifts), np.empty(n_shifts, dtype=np.int64))
        _jaccard_orientation(*args, *serial)
        _jaccard_orientation_parallel(
            *args,
            np.empty((len(lengths), n_shifts)),
            np.empty((len(lengths), n_shifts)),
            np.empty((len(lengths), n_shifts), dtype=np.int64),
            *parallel,
        )
        assert serial[0].tobytes() == parallel[0].tobytes()
        np.testing.assert_array_equal(serial[1], parallel[1])

    def test_empty_union_scores_zero(self):
        low = np.full(25, 0.5, dtype=np.float32)
        result = compare(ready_profile("q", [low]), ready_profile("t", [low]))
        assert result.score == 0.0
        assert result.n_sites == 0

    def test_rejects_different_row_counts(self):
        query = ready_profile("query", [[1.0, 2.0], [3.0, 4.0]])
        target = ready_profile("target", [[1.0, 2.0]])
        with pytest.raises(ValueError, match="same number of rows"):
            compare(query, target)

    def test_rejects_different_row_lengths(self):
        query = ready_profile("query", [[1.0, 2.0], [3.0]])
        target = ready_profile("target", [[1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="identical row lengths"):
            compare(query, target)


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

    def test_linear_preparation_is_unchanged_by_default(self, pwm_pair, batch):
        from mimosa import scan

        prepared = prepare_profile(pwm_pair[0], batch)
        raw = scan(pwm_pair[0], batch, strands="both")
        assert not prepared.circular
        np.testing.assert_array_equal(
            prepared.bundle.forward.offsets, raw.forward.offsets
        )

    def test_circular_preparation_keeps_one_score_per_base(self, batch):
        rng = np.random.default_rng(5)
        bamm = BaMM("bamm", rng.normal(size=(125, 6)).astype(np.float32), 2, 6)
        linear = prepare_profile(bamm, batch)
        circular = circular_profile(bamm, batch)

        assert linear.site_start_offset == 2
        assert circular.circular
        assert circular.site_start_offset == 0
        np.testing.assert_array_equal(circular.bundle.forward.offsets, batch.offsets)
        np.testing.assert_array_equal(circular.bundle.reverse.offsets, batch.offsets)
        assert circular != linear

    def test_circular_requires_zero_site_start_offset(self):
        scores = RaggedArray.from_rows([[1.0, 2.0, 3.0]])
        bundle = StrandPair(scores, scores)
        anchors = collect_both_anchors(bundle, 1.0, position_offset=1)
        with pytest.raises(ValueError, match="site_start_offset 0"):
            PreparedProfile(
                "bad", bundle, anchors, 1.0, site_start_offset=1, circular=True
            )

    def test_score_profile_is_only_marked_circular(self):
        scores = read_scores("examples/scores_1.fasta")
        linear = prepare_profile(scores, min_logerr=TAU0)
        circular = prepare_profile(scores, min_logerr=TAU0, circular=True)
        assert circular.circular and not linear.circular
        assert circular.bundle == linear.bundle

    @pytest.mark.parametrize("side", ("query", "target", "both"))
    def test_compare_rejects_linear_prepared_profiles(self, pwm_pair, batch, side):
        linear = prepare_profile(pwm_pair[0], batch, min_logerr=TAU0)
        circular = circular_profile(pwm_pair[1], batch)
        query = circular if side == "target" else linear
        target = circular if side == "query" else linear
        with pytest.raises(ValueError, match="circular"):
            compare(query, target)
        with pytest.raises(ValueError, match="circular"):
            compare_many(query, [target])

    def test_compare_rejects_linear_prepared_before_preparing_models(
        self, pwm_pair, batch, monkeypatch
    ):
        import importlib

        compare_module = importlib.import_module("mimosa.compare")
        linear = prepare_profile(pwm_pair[0], batch, min_logerr=TAU0)
        monkeypatch.setattr(
            compare_module,
            "_prepare_profile",
            lambda *args, **kwargs: pytest.fail("prepared a profile before validating"),
        )
        with pytest.raises(ValueError, match="circular"):
            compare(linear, pwm_pair[1], batch)
        with pytest.raises(ValueError, match="circular"):
            compare_many(pwm_pair[1], [linear], batch, min_logerr=TAU0)

    def test_equivalent_pwm_and_order_one_bamm_align_at_physical_zero(self):
        rng = np.random.default_rng(8)
        pwm_weights = rng.normal(size=(5, 4)).astype(np.float32)
        bamm_weights = np.empty((25, 4), dtype=np.float32)
        for code in range(25):
            bamm_weights[code] = pwm_weights[code % 5]

        pwm = PWM("pwm", pwm_weights, (0.25, 0.25, 0.25, 0.25))
        bamm = BaMM("bamm", bamm_weights, 1, 4)
        sequences = EncodedSequences.from_rows(
            [rng.integers(0, 4, size=200, dtype=np.uint8) for _ in range(50)]
        )

        for query, target in ((pwm, bamm), (bamm, pwm)):
            result = compare(
                query, target, sequences, search_range=0, min_logerr=TAU0
            )
            assert result.offset == 0
            assert result.score == pytest.approx(1.0)

    def test_compare_models_matches_reference(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        result = compare(m1, m2, batch, min_logerr=TAU0)
        assert isinstance(result, ComparisonResult)
        assert result.query == "MA0047.3"
        assert result.target == "MA0036.2"
        assert result.metric == "jaccard"
        assert result.orientation == "+-"
        assert result.offset == -2
        assert result.n_sites == 2828
        assert abs(float(result.score) - 0.37774196) < 1e-5

    def test_compare_score_profiles(self):
        s1 = read_scores("examples/scores_1.fasta")
        s2 = read_scores("examples/scores_2.fasta")
        result = compare(s1, s2, min_logerr=TAU0)
        assert result.offset == 4
        assert result.orientation == "++"
        assert result.n_sites == 3772
        assert abs(float(result.score) - 0.18173653) < 1e-5

    def test_default_threshold_is_four_and_warns_on_few_positions(
        self, pwm_pair, batch
    ):
        with pytest.warns(UserWarning, match="noisy"):
            result = compare(pwm_pair[0], pwm_pair[1], batch)
        with pytest.warns(UserWarning, match="noisy"):
            explicit = compare_many(
                pwm_pair[0], [pwm_pair[1]], batch, min_logerr=4.0
            )
        assert explicit == [result]

    def test_enough_positions_do_not_warn(self, pwm_pair, batch):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compare(pwm_pair[0], pwm_pair[1], batch, min_logerr=TAU0)

    @pytest.mark.parametrize("min_logerr", (0.0, -1.0, np.nan, np.inf))
    def test_compare_rejects_non_positive_threshold(self, pwm_pair, batch, min_logerr):
        with pytest.raises(ValueError, match="min_logerr"):
            compare(pwm_pair[0], pwm_pair[1], batch, min_logerr=min_logerr)
        with pytest.raises(ValueError, match="min_logerr"):
            compare_many(pwm_pair[0], [pwm_pair[1]], batch, min_logerr=min_logerr)

    def test_prepare_profile_defaults_to_positive_threshold(self, pwm_pair, batch):
        prepared = prepare_profile(pwm_pair[0], batch, circular=True)
        assert prepared.min_logerr == np.float32(4.0)

    @pytest.mark.parametrize("min_logerr", (0.0, -1.0, np.nan, np.inf))
    def test_prepare_profile_rejects_non_positive_threshold(
        self, pwm_pair, batch, min_logerr
    ):
        with pytest.raises(ValueError, match="min_logerr"):
            prepare_profile(pwm_pair[0], batch, min_logerr=min_logerr, circular=True)
        prepared = circular_profile(pwm_pair[0], batch)
        with pytest.raises(ValueError, match="min_logerr"):
            PreparedProfile(
                prepared.name,
                prepared.bundle,
                prepared.anchors,
                min_logerr,
                circular=True,
            )

    def test_anchor_collection_rejects_non_positive_threshold(self):
        scores = RaggedArray.from_rows([np.array([1.0, 5.0], dtype=np.float32)])
        with pytest.raises(ValueError, match="min_logerr"):
            collect_anchor_csr(scores, 0.0)

    def test_compare_self_is_perfect(self, pwm_pair, batch):
        m1, _ = pwm_pair
        result = compare(m1, m1, batch, min_logerr=TAU0)
        assert float(result.score) == 1.0
        assert result.offset == 0
        # Both strands align perfectly; the one with more sites wins the tie.
        assert result.orientation in ("++", "--")

    def test_reverse_complement_model_is_perfect_on_opposite_strands(
        self, pwm_pair, batch
    ):
        m1, _ = pwm_pair
        weights = np.array(m1.weights[:, ::-1])
        weights[:4] = weights[3::-1]
        reverse = PWM("reverse", weights, m1.background[::-1])
        result = compare(m1, reverse, batch, min_logerr=TAU0)
        assert float(result.score) == pytest.approx(1.0, abs=1e-3)
        assert result.offset == 0
        assert result.orientation in ("+-", "-+")

    @pytest.mark.parametrize("columns", (1, 3, 5))
    def test_shifted_model_recovers_the_column_shift(self, pwm_pair, batch, columns):
        from mimosa.profiles.alignment import _score_orientation_pair

        m1, _ = pwm_pair
        padding = np.zeros((5, columns), dtype=np.float32)
        shifted = PWM("shifted", np.hstack([padding, m1.weights]), m1.background)
        query = circular_profile(m1, batch)
        target = circular_profile(shifted, batch)

        score, shift, _ = _score_orientation_pair(
            query.bundle, target.bundle, query.anchors, target.anchors, 0, 0, 5, TAU0
        )
        assert (score, shift) == (1.0, -columns)
        # Site starts coincide on the reverse strand, which ties at shift 0.
        result = compare(query, target)
        assert float(result.score) == 1.0
        assert (result.orientation, result.offset) in (("++", -columns), ("--", 0))

    def test_compare_is_symmetric_up_to_mirrored_alignment(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        forward = compare(m1, m2, batch, min_logerr=TAU0)
        backward = compare(m2, m1, batch, min_logerr=TAU0)
        assert float(backward.score) == pytest.approx(float(forward.score), abs=1e-6)
        assert backward.offset == -forward.offset
        assert backward.orientation == forward.orientation[::-1]
        assert backward.n_sites == forward.n_sites

    def test_prepared_reuse(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = circular_profile(m1, batch)
        direct = compare(m1, m2, batch, min_logerr=TAU0)
        prepared = compare(query, m2, batch)
        assert prepared == direct

    def test_one_to_many(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = circular_profile(m1, batch)
        results = compare_many(query, [m2, m2], batch)
        assert len(results) == 2
        assert results[0] == results[1]

    def test_compare_many_preserves_target_order(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = circular_profile(m1, batch)
        target = circular_profile(m2, batch)
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
        query = circular_profile(pwm_pair[0], batch)
        target = circular_profile(pwm_pair[1], batch)
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
        query = circular_profile(pwm_pair[0], batch)
        target = circular_profile(pwm_pair[1], batch)
        monkeypatch.setattr(numba, "set_num_threads", calls.append)
        compare_module._prepare_and_compare_with_threads(
            query, target, batch, None, query.min_logerr, query.normalization,
            ProfileConfig(min_logerr=TAU0), None, None, 1,
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
            assert kwargs["circular"] is True
            return original(*args, **kwargs)

        previous_threads = numba.get_num_threads()
        monkeypatch.setattr(compare_module, "_prepare_profile", observed)
        compare_many(
            pwm_pair[0],
            [pwm_pair[1]],
            batch,
            min_logerr=TAU0,
            total_threads=2,
            inner_threads=2,
        )

        assert observed_threads == [2, 2]
        assert numba.get_num_threads() == previous_threads

    @pytest.mark.parametrize("threshold", (1.0, TAU0))
    def test_compare_many_joblib_matches_serial(
        self, pwm_pair, batch, tmp_path, threshold
    ):
        query = circular_profile(pwm_pair[0], batch, min_logerr=threshold)
        targets = [pwm_pair[1], pwm_pair[0], pwm_pair[1]]
        serial = compare_many(query, targets, batch, min_logerr=threshold)
        parallel = compare_many(
            query,
            targets,
            batch,
            min_logerr=threshold,
            cache=Cache(str(tmp_path)),
            total_threads=2,
            inner_threads=1,
        )
        assert parallel == serial

    def test_compare_many_joblib_cold_and_disk_cache_match(self, pwm_pair, batch, tmp_path):
        targets = [pwm_pair[1], pwm_pair[1]]
        expected = compare_many(pwm_pair[0], targets, batch, min_logerr=TAU0)
        cold = compare_many(
            pwm_pair[0], targets, batch, min_logerr=TAU0,
            cache=Cache(str(tmp_path)), total_threads=2,
        )
        disk = compare_many(
            pwm_pair[0], targets, batch, min_logerr=TAU0,
            cache=Cache(str(tmp_path)), total_threads=2,
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
        query = circular_profile(pwm_pair[0], batch)
        compare_many(query, [pwm_pair[1], pwm_pair[1]], batch)
        assert calls == 2

    def test_compare_many_reuses_sequence_context(self, pwm_pair, batch, tmp_path, monkeypatch):
        from mimosa import cache as cache_module

        calls = 0
        original = cache_module.sequence_fingerprint

        def counted(value):
            nonlocal calls
            calls += 1
            return original(value)

        monkeypatch.setattr(cache_module, "sequence_fingerprint", counted)
        cache = Cache(str(tmp_path))
        compare_many(pwm_pair[0], [pwm_pair[1]] * 3, batch, min_logerr=TAU0, cache=cache)
        assert calls == 1

    def test_compare_many_matches_direct(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = circular_profile(m1, batch)
        target = circular_profile(m2, batch)
        expected = compare(query, target, batch)
        assert compare_many(query, [target] * 2, batch) == [expected] * 2

    def test_compare_many_rejects_incompatible_prepared_target(self, pwm_pair, batch):
        query = circular_profile(pwm_pair[0], batch, min_logerr=1.0)
        target = circular_profile(pwm_pair[1], batch, min_logerr=TAU0)
        with pytest.raises(ValueError, match="different min_logerr"):
            compare_many(query, [target])

    def test_compare_many_prepared_shared_strands(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        prepared_target = circular_profile(m2, batch)
        shared_query = prepare_profile(
            ScoreProfile("shared-query", prepared_target.bundle.forward),
            min_logerr=TAU0,
            circular=True,
        )
        prepared_query = circular_profile(m1, batch)
        shared_target = prepare_profile(
            ScoreProfile("shared-target", prepared_query.bundle.forward),
            min_logerr=TAU0,
            circular=True,
        )
        for query, targets, orientations in (
            (shared_query, [prepared_target] * 2, ("++", "+-")),
            (prepared_query, [shared_target] * 2, ("++", "-+")),
        ):
            serial = [compare(query, target) for target in targets]
            results = compare_many(query, targets, batch)
            assert results == serial
            assert all(result.orientation in orientations for result in results)

    @pytest.mark.parametrize("threshold", (1.0, TAU0))
    def test_alignment_row_parallel_matches_serial(
        self, pwm_pair, batch, monkeypatch, threshold
    ):
        import importlib

        alignment_module = importlib.import_module("mimosa.profiles.alignment")
        query = circular_profile(pwm_pair[0], batch, min_logerr=threshold)
        target = circular_profile(pwm_pair[1], batch, min_logerr=threshold)
        monkeypatch.setattr(
            alignment_module, "use_alignment_parallel", lambda *args, **kwargs: False
        )
        serial = compare(query, target)
        monkeypatch.setattr(
            alignment_module, "use_alignment_parallel", lambda *args, **kwargs: True
        )
        parallel = compare(query, target)
        assert parallel == serial

    def test_prepared_profile_picklable(self, pwm_pair, batch):
        import pickle

        m1, _ = pwm_pair
        for prepared in (prepare_profile(m1, batch), circular_profile(m1, batch)):
            roundtrip = pickle.loads(pickle.dumps(prepared))
            assert roundtrip == prepared
            assert roundtrip.circular == prepared.circular

    def test_prepared_profile_rejects_negative_anchor(self, pwm_pair, batch):
        from mimosa.profiles.anchors import AnchorCSR

        prepared = prepare_profile(pwm_pair[0], batch)
        bad = AnchorCSR(np.array([-1], dtype=np.int64), np.array([0, 1], dtype=np.int64))
        with pytest.raises(ValueError):
            PreparedProfile(
                prepared.name,
                StrandPair(RaggedArray.from_rows([[1.0]]), RaggedArray.from_rows([[1.0]])),
                (bad, bad),
            )

    def test_metric_is_not_selectable(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        with pytest.raises(TypeError):
            compare(m1, m2, batch, metric="jaccard")
        with pytest.raises(TypeError):
            compare_many(m1, [m2], batch, metric="jaccard")

    def test_prepared_threshold_mismatch(self, pwm_pair, batch):
        m1, m2 = pwm_pair
        query = circular_profile(m1, batch, min_logerr=1.0)
        with pytest.raises(ValueError):
            compare(query, m2, batch, min_logerr=2.0)

    def test_prepared_normalization_mismatch_is_rejected(self, pwm_pair, batch):
        query = circular_profile(pwm_pair[0], batch)
        target = circular_profile(pwm_pair[1], batch)

        with pytest.raises(ValueError, match="requested normalization"):
            compare(query, target, normalization=EmpiricalLogTail())

    def test_result_dict_includes_metric_and_zero_contributing_sites(self):
        result = ComparisonResult("query", "target", np.float32(0.0), 0, "++")
        assert result.to_dict()["n_sites"] == 0
        assert result.to_dict()["metric"] == "jaccard"

    def test_mixed_unsupported(self, pwm_pair, batch):
        m1, _ = pwm_pair
        s1 = read_scores("examples/scores_1.fasta")
        with pytest.raises(ValueError):
            compare(s1, m1, batch)


class TestProfileConfig:
    def test_config_defaults(self):
        config = ProfileConfig()
        assert config.search_range == 5
        assert config.min_logerr == 4.0

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            ProfileConfig(search_range=-1)
        for min_logerr in (np.nan, np.inf, 0.0, -2.0):
            with pytest.raises(ValueError):
                ProfileConfig(min_logerr=min_logerr)
        with pytest.raises(TypeError):
            ProfileConfig(metric="jaccard")


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
