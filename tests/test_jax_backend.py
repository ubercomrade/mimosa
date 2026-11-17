"""Parity tests for the experimental, optional JAX primitives."""
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from mimosa import EncodedSequences, SiteGA, compare, prepare_profile, read_model, relabel_model  # noqa: E402
from mimosa._jax_backend import compare_circular, normalize_exact, scan_circular  # noqa: E402
from mimosa._jax_monte_carlo import direct_mc  # noqa: E402
from mimosa.profiles.normalization import (  # noqa: E402
    EmpiricalLogTail, _fit_exact, normalize_bundle,
)
from mimosa.scan import _scan_batch_into  # noqa: E402


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
    expected = _scan_batch_into(model, sequences, "both", circular=True)
    actual = scan_circular(model, sequences)
    for reference, result in zip((expected.forward.data, expected.reverse.data), actual):
        np.testing.assert_array_equal(np.asarray(result), reference)


@pytest.mark.parametrize("background", [False, True])
def test_exact_normalization_matches_cpu(background):
    model = read_model("examples/myog.ihbcp", order=4)
    sequences = EncodedSequences.from_strings(["ACGTNACGTN" * 5, "", "TTTTACGT" * 3])
    other = EncodedSequences.from_strings(["TGCATGCAN" * 7, "AC" * 20])
    raw = _scan_batch_into(model, sequences, "both", circular=True)
    calibration = _scan_batch_into(model, other, "both", circular=True) if background else raw
    reference = normalize_bundle(_fit_exact(EmpiricalLogTail(), calibration), raw)
    scores = np.concatenate((raw.forward.data, raw.reverse.data))
    cal = np.concatenate((calibration.forward.data, calibration.reverse.data))
    expected = np.concatenate((reference.forward.data, reference.reverse.data))
    np.testing.assert_array_equal(np.asarray(normalize_exact(scores, cal)), expected)


def test_normalization_boundaries_ties_and_empty():
    values = np.array([-100, -1, 0, 0.5, 1, 2, 100], np.float32)
    calibration = np.array([-1, -1, 0, 1, 1], np.float32)
    expected = np.array([0, 0, -np.log10(3 / 5), -np.log10(2 / 5),
                         -np.log10(2 / 5), -np.log10(2 / 5), -np.log10(2 / 5)],
                        np.float32)
    np.testing.assert_array_equal(np.asarray(normalize_exact(values, calibration)), expected)
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
    one = direct_mc([q], [t], sequences, **options)
    many = direct_mc([extra, q], [extra, t], sequences, **options)
    for first, matrix in zip(one, many):
        assert first[0, 0] == matrix[1, 1]


def test_direct_mc_counts_identity_ties(monkeypatch):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 20, "TCGA" * 20])
    monkeypatch.setattr(
        "mimosa._jax_monte_carlo._permutations",
        lambda model, *args: np.tile(np.arange(4), (model.motif_length, 1)),
    )
    scores, counts, pvalues = direct_mc(
        [model], [model], sequences, k=2, seed=9, min_logerr=0.01, search_range=1,
    )
    assert scores[0, 0] == 1
    assert counts[0, 0] == 2
    assert pvalues[0, 0] == 1


@pytest.mark.parametrize("path,order,group", [
    ("examples/foxa2.meme", None, "full"),
    ("examples/myog.ihbcp", 4, "gc"),
    ("tests/fixtures/stat_dimont-model-1.xml", None, "full"),
    ("tests/fixtures/slim/example-model-1.xml", None, "gc"),
])
def test_direct_mc_matches_cpu_reference(path, order, group):
    from benchmarks.direct_mc_prototype import direct_mc as cpu_direct_mc

    q = read_model("examples/gata4.meme")
    t = read_model(path, order=order)
    rows = np.random.default_rng(19).integers(0, 5, (4, 80), dtype=np.uint8)
    sequences = EncodedSequences.from_rows(rows)
    options = dict(k=2, seed=12, group=group, min_logerr=0.5, search_range=1)
    expected = cpu_direct_mc([q], [t], sequences, **options)
    actual = direct_mc([q], [t], sequences, **options)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
