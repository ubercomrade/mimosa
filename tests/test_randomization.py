"""Independent f(pi(x)) checks and end-to-end calibrated-null contracts."""
import hashlib
from contextlib import nullcontext

import numpy as np
import pytest

from mimosa import (
    EncodedSequences, InputRelabeledModel, WindowModel,
    alphabet_permutations, prepare_profile,
    read_model, relabel_model, scan,
)
from mimosa.cache import Cache
from mimosa.errors import ModelFormatError, ModelInterfaceError
from mimosa.io.bundles import model_fingerprint
from mimosa.models import window_size
from mimosa.scan import _scan_batch


@pytest.fixture(params=[
    ("examples/foxa2.meme", None), ("examples/myog.ihbcp", 0),
    ("examples/myog.ihbcp", 4),
    pytest.param(("examples/sitega_gata2.mat", None),
                 marks=pytest.mark.skip(reason="SiteGA is disabled")),
    ("tests/fixtures/stat_dimont-model-1.xml", None),
    ("tests/fixtures/slim/example-model-1.xml", None),
])
def model(request):
    path, order = request.param
    return read_model(path, order=order)


def window_function(model):
    width = window_size(model)
    def evaluate(windows):
        batch = EncodedSequences(windows.reshape(-1), np.arange(len(windows)+1)*width)
        return scan(model, batch, strands="forward").data
    return evaluate


def direct_recoded_reference(model, batch, permutations, circular):
    """Explicit per-window implementation, independent of wrapper/native maps."""
    width = window_size(model)
    forward, reverse = [], []
    for row in range(len(batch)):
        dna = batch[row]
        n = len(dna) if circular else max(0, len(dna)-width+1)
        fwd, rev = [], []
        for start in range(n):
            if circular:
                word = dna[(start-model.left_context+np.arange(width)) % len(dna)]
            else:
                word = dna[start:start+width]
            for output, oriented in ((fwd, word), (rev, np.where(word[::-1] < 4, 3-word[::-1], 4))):
                changed = np.array(oriented, dtype=np.uint8, copy=True)
                for j in range(model.motif_length):
                    coordinate = model.left_context+j
                    if changed[coordinate] < 4:
                        changed[coordinate] = permutations[j, changed[coordinate]]
                encoded = EncodedSequences(changed, np.array([0, width]))
                output.append(scan(model, encoded, strands="forward").data[0])
        forward.extend(fwd)
        reverse.extend(rev)
    return np.array(forward, np.float32), np.array(reverse, np.float32)


@pytest.mark.parametrize("circular", [False, True])
def test_native_and_generic_equal_explicit_input_recoding(model, circular):
    rng = np.random.default_rng(83)
    width = window_size(model)
    batch = EncodedSequences.from_rows([
        rng.integers(0, 5, size=width+3, dtype=np.uint8), np.array([], np.uint8),
        rng.integers(0, 5, size=2, dtype=np.uint8),
        rng.integers(0, 5, size=width-1, dtype=np.uint8),
    ])
    permutations = alphabet_permutations(model.motif_length, 112)
    native = relabel_model(model, permutations=permutations)
    wrapped = InputRelabeledModel(model, permutations, batch_size=7)
    a = _scan_batch(native, batch, "both", circular)
    b = _scan_batch(wrapped, batch, "both", circular)
    expected = direct_recoded_reference(model, batch, permutations, circular)
    for x, y, z in zip((a.forward, a.reverse), (b.forward, b.reverse), expected):
        np.testing.assert_array_equal(x.offsets, y.offsets)
        np.testing.assert_array_equal(x.data, y.data)
        np.testing.assert_array_equal(x.data, z)


def test_identity_native_and_generic(model):
    permutations = np.tile(np.arange(4), (model.motif_length, 1))
    native = relabel_model(model, permutations=permutations)
    np.testing.assert_array_equal(native.weights, model.weights)
    width = window_size(model)
    word = np.random.default_rng(3).integers(0, 4, size=width+2, dtype=np.uint8)
    actual = scan(InputRelabeledModel(model, permutations),
                  EncodedSequences(word, np.array([0, len(word)])), strands="both")
    reference = scan(model, EncodedSequences(word, np.array([0, len(word)])), strands="both")
    np.testing.assert_array_equal(actual.forward.data, reference.forward.data)
    np.testing.assert_array_equal(actual.reverse.data, reference.reverse.data)


def test_full_window_bijection_preserves_scalar_score_multiset():
    # Nonlinear in all positions; no PWM/BaMM assumptions.
    words = np.indices((4, 4, 4)).reshape(3, -1).T.astype(np.uint8)
    def score(x):
        return (x[:, 0] == x[:, 1]).astype(np.float32)*4 + (x[:, 1]*x[:, 2]).astype(np.float32)
    source = WindowModel("nonlinear", 3, score, fingerprint="test-nonlinear-v1", calibration_group="toy-v1")
    changed = relabel_model(source, seed=10)
    encoded = EncodedSequences(words.reshape(-1), np.arange(len(words)+1)*3)
    np.testing.assert_array_equal(np.sort(scan(source, encoded).data), np.sort(scan(changed, encoded).data))


@pytest.mark.parametrize("policy", ["forward", "reverse", "best", "both"])
def test_window_batch_size_and_strand_policies(policy):
    def score(x):
        return (x[:, 0]+3*x[:, 1]+x[:, -1]).astype(np.float32)
    a = WindowModel("w", 2, score, left_context=1, right_context=2, batch_size=1)
    b = WindowModel("w", 2, score, left_context=1, right_context=2, batch_size=100)
    batch = EncodedSequences.from_strings(["ACGTNACG", "", "N", "TTTTTT", "CGTAC"])
    x, y = scan(a, batch, strands=policy), scan(b, batch, strands=policy)
    pairs = [(x.forward, y.forward), (x.reverse, y.reverse)] if policy == "both" else [(x, y)]
    for u, v in pairs:
        np.testing.assert_array_equal(u.data, v.data)
        np.testing.assert_array_equal(u.offsets, v.offsets)


def _invalid_window_callback(kind):
    def callback(x):
        if kind == "short":
            return np.ones((len(x), 1))
        if kind == "long":
            return np.ones(len(x) + 1)
        if kind == "nan":
            return np.full(len(x), np.nan)
        if kind == "inf":
            return np.full(len(x), np.inf)
        if kind == "large":
            return np.full(len(x), 1e100)
        return np.full(len(x), 1 + 2j)
    return callback


@pytest.mark.parametrize("kind,expected", [
    ("short", ModelInterfaceError), ("long", ModelInterfaceError),
    ("nan", ModelFormatError), ("inf", ModelFormatError),
    ("large", ModelFormatError), ("complex", ModelInterfaceError),
])
def test_invalid_window_outputs_are_rejected(kind, expected):
    # The large callback deliberately exceeds Float32 range during JAX tracing.
    warning = (
        pytest.warns(RuntimeWarning, match="overflow encountered in cast")
        if kind == "large" else nullcontext()
    )
    with warning, pytest.raises(expected, match="scan"):
        scan(
            WindowModel("invalid", 2, _invalid_window_callback(kind)),
            EncodedSequences.from_strings(["ACGT"]),
        )


def test_permutation_copy_freeze_and_fingerprints():
    source = read_model("examples/foxa2.meme")
    permutations = alphabet_permutations(source.motif_length, 12).copy()
    a = InputRelabeledModel(source, permutations)
    old = model_fingerprint(a)
    permutations[0] = permutations[0, ::-1]
    assert model_fingerprint(a) == old
    with pytest.raises(ValueError):
        a.permutations[0, 0] = 2
    with pytest.raises(AttributeError):
        a.permutations = permutations
    b = InputRelabeledModel(source, alphabet_permutations(source.motif_length, 13))
    assert model_fingerprint(b) != old


@pytest.mark.parametrize("permutations", [np.zeros((2, 4), int), np.ones((1, 4), bool), [[0, 1, 2, 4]]])
def test_invalid_permutations(permutations):
    source = WindowModel("test", 1, lambda x: x[:, 0])
    with pytest.raises(ValueError, match="permutation"):
        relabel_model(source, permutations=permutations)


@pytest.fixture
def null_batch():
    rng = np.random.default_rng(111)
    return EncodedSequences(rng.integers(0, 4, size=25600, dtype=np.uint8), np.arange(257)*100)


@pytest.fixture
def bamm_models():
    return [read_model(f"examples/{name}.ihbcp", order=0) for name in ("myog", "gata2")]


def test_relabeled_profile_cache_reuses_normalization(null_batch, tmp_path):
    source = read_model("examples/myog.ihbcp", order=4)
    model = InputRelabeledModel(source, alphabet_permutations(source.motif_length, 5))
    cache = Cache(tmp_path/"cache")
    a = prepare_profile(model, null_batch, min_logerr=2, circular=True, cache=cache)
    b = prepare_profile(model, null_batch, min_logerr=1, circular=True, cache=cache)
    np.testing.assert_array_equal(a.bundle.forward.data, b.bundle.forward.data)
    assert a.anchors[0].positions.size < b.anchors[0].positions.size


def test_declared_function_fingerprint_changes_with_configuration():
    identity = hashlib.sha256(b"weights-v1").hexdigest()
    a = WindowModel("m", 3, lambda x: x.sum(axis=1), fingerprint=identity, calibration_group="toy-v1")
    b = WindowModel("m", 4, lambda x: x.sum(axis=1), fingerprint=identity, calibration_group="toy-v1")
    assert model_fingerprint(a) != model_fingerprint(b)
