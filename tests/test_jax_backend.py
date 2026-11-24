"""Parity tests for the experimental, optional JAX primitives."""
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from mimosa import (  # noqa: E402
    EncodedSequences, SiteGA, compare, prepare_profile, read_model, relabel_model, window_size,
)
from mimosa.backend import compare_circular, normalize_exact, scan_circular, scan_linear  # noqa: E402
from mimosa.monte_carlo import direct_mc  # noqa: E402
from mimosa.profiles.normalization import (  # noqa: E402
    EmpiricalLogTail, fit, normalize_bundle,
)
from mimosa.scan import _scan_batch  # noqa: E402


@pytest.fixture(autouse=True)
def _enable_x64_for_low_level_tests():
    with jax.enable_x64(True):
        yield


def _legacy_scan(model, sequences, circular):
    forward, reverse, offsets = [], [], [0]
    for row_index in range(len(sequences)):
        row = sequences[row_index]
        count = len(row) if circular else max(len(row) - window_size(model) + 1, 0)
        if circular and len(row):
            left = model.left_context
            right = window_size(model) - left - 1
            row = row[np.arange(-left, len(row) + right) % len(row)]
        f, r = np.empty(count, np.float32), np.empty(count, np.float32)
        order = getattr(model, "order", 0)
        width = window_size(model)
        for pos in range(count):
            window = row[pos:pos + width]
            rc = np.where(window[::-1] == 4, 4, 3 - window[::-1])
            for output, oriented in ((f, window), (r, rc)):
                total = np.float32(0)
                for term in range(model.motif_length):
                    code = 0
                    for base in oriented[term:term + order + 1]:
                        code = 5 * code + int(base)
                    total = np.float32(total + model.weights[code, term])
                output[pos] = total
        forward.extend(f)
        reverse.extend(r)
        offsets.append(offsets[-1] + count)
    return np.asarray(forward, np.float32), np.asarray(reverse, np.float32), np.asarray(offsets)


@pytest.mark.parametrize("path,order", [
    ("examples/foxa2.meme", None),
    ("examples/myog.ihbcp", 0),
    ("examples/myog.ihbcp", 4),
    ("tests/fixtures/stat_dimont-model-1.xml", None),
    ("tests/fixtures/slim/example-model-1.xml", None),
])
@pytest.mark.parametrize("relabeled", [False, True])
def test_ragged_circular_scan_matches_cpu(path, order, relabeled):
    model = read_model(path, order=order)
    if relabeled:
        model = relabel_model(model, seed=117)
    sequences = EncodedSequences.from_strings(["", "N", "AC", "ACGTNNGTA", "TGCATGCA" * 5])
    fwd, rev, _ = _legacy_scan(model, sequences, circular=True)
    actual = scan_circular(model, sequences)
    for reference, result in zip((fwd, rev), actual):
        np.testing.assert_array_equal(np.asarray(result), reference)


@pytest.mark.parametrize("path,order", [
    ("examples/foxa2.meme", None),
    ("examples/myog.ihbcp", 0),
    ("examples/myog.ihbcp", 4),
    ("tests/fixtures/stat_dimont-model-1.xml", None),
    ("tests/fixtures/slim/example-model-1.xml", None),
])
@pytest.mark.parametrize("relabeled", [False, True])
def test_ragged_linear_scan_matches_cpu(path, order, relabeled):
    model = read_model(path, order=order)
    if relabeled:
        model = relabel_model(model, seed=117)
    sequences = EncodedSequences.from_strings(["", "N", "AC", "ACGTNNGTA", "TGCATGCA" * 5])
    fwd, rev, expected_offsets = _legacy_scan(model, sequences, circular=False)
    actual, offsets = scan_linear(model, sequences)
    np.testing.assert_array_equal(offsets, expected_offsets)
    for reference, result in zip((fwd, rev), actual):
        np.testing.assert_array_equal(np.asarray(result), reference)


@pytest.mark.parametrize("background", [False, True])
def test_exact_normalization_matches_cpu(background):
    model = read_model("examples/myog.ihbcp", order=4)
    sequences = EncodedSequences.from_strings(["ACGTNACGTN" * 5, "", "TTTTACGT" * 3])
    other = EncodedSequences.from_strings(["TGCATGCAN" * 7, "AC" * 20])
    raw = _scan_batch(model, sequences, "both", circular=True)
    calibration = _scan_batch(model, other, "both", circular=True) if background else raw
    calibration_scores = np.maximum(calibration.forward.data, calibration.reverse.data)
    reference = normalize_bundle(fit(EmpiricalLogTail(), calibration_scores), raw)
    scores = np.concatenate((raw.forward.data, raw.reverse.data))
    cal = calibration_scores
    expected = np.concatenate((reference.forward.data, reference.reverse.data))
    np.testing.assert_array_equal(np.asarray(normalize_exact(scores, cal)), expected)


def test_position_calibration_counts_unique_positions():
    from mimosa import RaggedArray, StrandPair
    from mimosa.profiles.prepared import _normalize_with_jax

    forward = RaggedArray.from_rows([np.arange(1000, dtype=np.float32)])
    reverse = RaggedArray.from_rows([np.arange(1000, dtype=np.float32)[::-1]])
    raw = StrandPair(forward, reverse)
    actual = _normalize_with_jax(raw)
    maxima = np.maximum(forward.data, reverse.data)
    expected = normalize_bundle(fit(EmpiricalLogTail(), maxima), raw)
    np.testing.assert_array_equal(actual.forward.data, expected.forward.data)
    np.testing.assert_array_equal(actual.reverse.data, expected.reverse.data)
    assert np.count_nonzero(np.maximum(actual.forward.data, actual.reverse.data) >= 2) == 10


def test_normalization_boundaries_ties_and_empty():
    values = np.array([-100, -1, -0.5, -0.0, 0.0, 0.5, 1, 2, 100], np.float32)
    calibration = np.array([-1, -1, -0.0, 0.0, 1, 1], np.float32)
    expected = -np.log10(np.array([6, 6, 4, 4, 4, 2, 2, 2, 2], np.float64) / 6)
    actual = np.asarray(normalize_exact(values, calibration))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.astype(np.float32).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(normalize_exact(values, np.array([], np.float32))),
                                  np.zeros_like(values))


def test_unsupported_model_is_not_silently_fallback():
    model = object.__new__(SiteGA)  # Rejected before any disabled model fields are read.
    with pytest.raises(TypeError, match="supports PWM"):
        scan_circular(model, EncodedSequences.from_strings(["ACGT"]))


def test_empty_circular_batch():
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["", ""])
    result = scan_circular(model, sequences)
    assert [np.asarray(a).size for a in result] == [0, 0]


def test_public_preparation_does_not_change_jax_x64_configuration():
    script = """
import jax
from mimosa import EncodedSequences, prepare_profile, read_model
assert not jax.config.x64_enabled
batch = EncodedSequences.from_strings(["ACGTN" * 8])
prepare_profile(read_model("examples/foxa2.meme"), batch, min_logerr=2, circular=True)
assert not jax.config.x64_enabled
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="false")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_public_monte_carlo_preserves_x64_and_pair_independence():
    script = """
import jax
import numpy as np
from mimosa import EncodedSequences, direct_mc, read_model
q = read_model("examples/foxa2.meme")
t = read_model("examples/gata4.meme")
s = EncodedSequences.from_strings(["ACGTN" * 12, "", "TGCAT" * 10])
b = EncodedSequences.from_strings(["TGCA" * 20])
assert not jax.config.x64_enabled
for group in ("full", "gc"):
    kw = dict(k=2, seed=19, group=group, min_logerr=1, search_range=1, background=b)
    single = direct_mc(q, [t], s, **kw)
    many = direct_mc(q, [q, t], s, **kw)
    for a, m in zip(single, many):
        assert a[0, 0] == m[0, 1]
    assert single[2][0, 0] == (single[1][0, 0] + 1) / 3
assert not jax.config.x64_enabled
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="false")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


@pytest.mark.parametrize("search_range", [0, 2, 7])
def test_comparison_ragged_and_orientation(search_range):
    q = read_model("examples/foxa2.meme")
    t = read_model("examples/gata2.ihbcp", order=4)
    sequences = EncodedSequences.from_strings([
        "ACGTN" * 13, "", "T", "NNACGT" * 7, "TAGCGATC" * 11,
    ])
    qp = prepare_profile(q, sequences, min_logerr=1, circular=True)
    tp = prepare_profile(t, sequences, min_logerr=1, circular=True)
    with pytest.warns(UserWarning):
        expected = compare(qp, tp, search_range=search_range)
    query = np.stack((qp.bundle.forward.data, qp.bundle.reverse.data))
    target = np.stack((tp.bundle.forward.data, tp.bundle.reverse.data))
    score, shift, orientation, sites = map(np.asarray, compare_circular(
        query, target, sequences.offsets, min_logerr=1, search_range=search_range,
    ))
    assert score == expected.score
    assert shift == expected.offset
    assert ("++", "+-", "-+", "--")[int(orientation)] == expected.orientation
    assert sites == expected.n_sites


def test_comparison_empty_and_invalid_layout():
    empty = np.empty((2, 0), np.float32)
    result = compare_circular(empty, empty, [0, 0], min_logerr=2)
    assert tuple(map(int, result)) == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="offsets"):
        compare_circular(np.ones((2, 2)), np.ones((2, 2)), [0, 3, 2], min_logerr=2)
    with pytest.raises(ValueError, match="offsets"):
        compare_circular(np.ones((2, 2)), np.ones((2, 2)),
                         np.array([0, 3, 2], dtype=np.uint64), min_logerr=2)


@pytest.mark.parametrize("group", ["full", "gc"])
def test_direct_mc_pair_independent_of_collection(group):
    q = read_model("examples/foxa2.meme")
    t = read_model("tests/fixtures/stat_dimont-model-1.xml")
    extra = read_model("examples/myog.ihbcp", order=0)
    sequences = EncodedSequences.from_strings(["ACGTNNAGTC" * 5, "", "GCTAT" * 11])
    options = dict(k=2, seed=42, group=group, min_logerr=1, search_range=1)
    one = direct_mc(q, [t], sequences, **options)
    many = direct_mc(q, [extra, t], sequences, **options)
    for first, matrix in zip(one, many):
        assert first[0, 0] == matrix[0, 1]


def test_direct_mc_counts_identity_ties(monkeypatch):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 20, "TCGA" * 20])
    monkeypatch.setattr(
        "mimosa.monte_carlo._permutations",
        lambda model, *args: np.tile(np.arange(4), (model.motif_length, 1)),
    )
    scores, counts, pvalues = direct_mc(
        model, [model], sequences, k=2, seed=9, min_logerr=0.01, search_range=1,
    )
    assert scores[0, 0] == 1
    assert counts[0, 0] == 2
    assert pvalues[0, 0] == 1


@pytest.mark.parametrize("search_range", [0, 2, 5])
@pytest.mark.parametrize("size", [1, 3, 8])
def test_batched_exceedances_match_scalar_scores(search_range, size):
    import jax.numpy as jnp

    from mimosa.monte_carlo import _count_exceedances

    rng = np.random.default_rng(89)
    offsets = np.array([0, 27, 27, 74, 101], dtype=np.int64)
    lengths = np.diff(offsets)
    starts = jnp.asarray(np.repeat(offsets[:-1], lengths).astype(np.int32))
    sizes = jnp.asarray(np.repeat(lengths, lengths).astype(np.int32))
    target = rng.uniform(0, 4, (2, offsets[-1])).astype(np.float32)
    queries = rng.uniform(0, 4, (size, 2, offsets[-1])).astype(np.float32)
    scores = np.array([
        compare_circular(query, target, offsets, min_logerr=2, search_range=search_range)[0]
        for query in queries
    ])
    # Using an actual score as the threshold also checks an exact tie.
    observed = scores[0]
    actual = _count_exceedances(jnp.asarray(queries), jnp.asarray(target), starts, sizes,
                                np.float32(2), observed, search_range=search_range)
    assert int(actual) == np.count_nonzero(scores >= observed)


@pytest.mark.parametrize("search_range", [0, 2, 5])
def test_sparse_exceedances_match_dense_near_decision_boundary(search_range):
    import jax.numpy as jnp

    from mimosa.profiles.sparse import count_exceedances, make_anchors

    rng = np.random.default_rng(640)
    offsets = np.array([0, 1, 1, 8, 39, 296], dtype=np.int64)
    lengths = np.diff(offsets)
    starts = jnp.asarray(np.repeat(offsets[:-1], lengths).astype(np.int32))
    sizes = jnp.asarray(np.repeat(lengths, lengths).astype(np.int32))
    queries = rng.uniform(0, 2.9, (8, 2, offsets[-1])).astype(np.float32)
    target = rng.uniform(0, 2.9, (2, offsets[-1])).astype(np.float32)
    for query in queries:
        for strand in query:
            strand[rng.choice(offsets[-1], 5, replace=False)] = rng.uniform(3, 6, 5)
    for strand in target:
        strand[rng.choice(offsets[-1], 5, replace=False)] = rng.uniform(3, 6, 5)
    q = jnp.asarray(queries)
    t = jnp.asarray(target)
    qanchors = make_anchors(q, np.float32(3))
    tanchors = make_anchors(t, np.float32(3))
    assert qanchors is not None and tanchors is not None
    dense = np.array([
        compare_circular(query, target, offsets, min_logerr=3, search_range=search_range)[0]
        for query in queries
    ])
    boundary = dense[0]
    for observed in (np.float32(0), boundary,
                     np.nextafter(boundary, np.float32(0)),
                     np.nextafter(boundary, np.float32(1)), np.float32(1)):
        actual = count_exceedances(
            q, t, *qanchors, *tanchors, starts, sizes, np.float32(3), observed,
            search_range=search_range,
        )
        assert int(actual) == np.count_nonzero(dense >= observed)


def test_sparse_near_tie_runs_dense_fallback(monkeypatch):
    import jax.numpy as jnp
    from mimosa.profiles import sparse

    def near_below(*args, **kwargs):
        return jnp.nextafter(jnp.float32(0.5), jnp.float32(0))

    def dense_tie(*args, **kwargs):
        return jnp.float32(0.5), jnp.int32(0), jnp.int32(0), jnp.int64(0)

    sparse.count_exceedances.clear_cache()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sparse, "_sparse_score", near_below)
            patch.setattr(sparse, "_compare_circular", dense_tie)
            result = sparse.count_exceedances(
                jnp.zeros((1, 2, 4)), jnp.zeros((2, 4)),
                jnp.zeros((1, 2, 1), jnp.int32), jnp.ones((1, 2, 1), bool),
                jnp.zeros((2, 1), jnp.int32), jnp.ones((2, 1), bool),
                jnp.zeros(4, jnp.int32), jnp.full(4, 4, jnp.int32),
                jnp.float32(1), jnp.float32(0.5), search_range=0,
            )
            assert int(result) == 1
    finally:
        sparse.count_exceedances.clear_cache()


@pytest.mark.parametrize("path,order,group", [
    ("examples/foxa2.meme", None, "full"),
    ("examples/myog.ihbcp", 4, "gc"),
    ("tests/fixtures/stat_dimont-model-1.xml", None, "full"),
    ("tests/fixtures/slim/example-model-1.xml", None, "gc"),
])
def test_direct_mc_matches_cpu_reference(path, order, group):
    from mimosa.monte_carlo import _permutations
    from mimosa.randomization import relabel_model

    q = read_model(path, order=order)
    t = read_model("examples/gata4.meme")
    rows = np.random.default_rng(19).integers(0, 5, (4, 80), dtype=np.uint8)
    sequences = EncodedSequences.from_rows(rows)
    options = dict(k=2, seed=12, group=group, min_logerr=0.5, search_range=1)
    observed = compare(q, t, sequences, min_logerr=0.5, search_range=1).score
    null = [
        compare(relabel_model(q, permutations=_permutations(q, 12, i, group)), t,
                sequences, min_logerr=0.5, search_range=1).score
        for i in range(2)
    ]
    actual = direct_mc(q, [t], sequences, **options)
    assert actual[0][0, 0] == observed
    assert actual[1][0, 0] == sum(x >= observed for x in null)
