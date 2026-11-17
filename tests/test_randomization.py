"""Independent f(pi(x)) checks and end-to-end calibrated-null contracts."""
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mimosa import (
    EncodedSequences, InputRelabeledModel, NullDistribution, WindowModel,
    alphabet_permutations, annotate_results, build_null, compare, prepare_profile,
    read_model, relabel_model, scan,
)
from mimosa.cache import Cache
from mimosa.errors import MimosaError, ModelFormatError, ModelInterfaceError
from mimosa.io.bundles import model_fingerprint, read_null_bundle, write_null_bundle
from mimosa.models import window_size
from mimosa.scan import _scan_batch_into
from mimosa.statistics import check_null_compatibility


@pytest.fixture(params=[
    ("examples/foxa2.meme", None), ("examples/myog.ihbcp", 0),
    ("examples/myog.ihbcp", 4), ("examples/sitega_gata2.mat", None),
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
    a = _scan_batch_into(native, batch, "both", circular)
    b = _scan_batch_into(wrapped, batch, "both", circular)
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
    forward, reverse = np.empty(3, np.float32), np.empty(3, np.float32)
    InputRelabeledModel(model, permutations).scan_into(word, forward, reverse)
    reference = scan(model, EncodedSequences(word, np.array([0, len(word)])), strands="both")
    np.testing.assert_array_equal(forward, reference.forward.data)
    np.testing.assert_array_equal(reverse, reference.reverse.data)


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


@pytest.mark.parametrize("callback", [
    lambda x: np.ones((len(x), 1)), lambda x: np.ones(len(x)+1),
    lambda x: np.full(len(x), np.nan), lambda x: np.full(len(x), np.inf),
    lambda x: np.full(len(x), 1e100), lambda x: np.full(len(x), 1+2j),
])
def test_invalid_window_outputs_are_rejected(callback):
    with pytest.raises(ModelInterfaceError):
        scan(WindowModel("invalid", 2, callback), EncodedSequences.from_strings(["ACGT"]))


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


def test_null_replay_bundle_and_compatibility(null_batch, bamm_models, tmp_path):
    options = dict(sequences=null_batch, n_samples=24, seed=112, min_logerr=2, randomization="input-relabel")
    a, b = build_null(bamm_models, **options), build_null(bamm_models, **options)
    np.testing.assert_array_equal(a.raw_scores, b.raw_scores)
    assert a.pairs == b.pairs
    assert a.strategy == "profile-input-relabel"
    assert a.contract["model_group"] == "BaMM:order=0"
    assert isinstance(a.contract["model_seeds"], tuple)
    path = tmp_path/"null"
    write_null_bundle(path, a)
    stored = NullDistribution(**read_null_bundle(path))
    assert dict(stored.contract) == dict(a.contract)
    assert stored.pairs == a.pairs
    check_null_compatibility(stored, sequences=null_batch, background=None, search_range=5,
                             min_logerr=2, models=bamm_models)
    with pytest.raises(MimosaError, match="compared models"):
        check_null_compatibility(stored, sequences=null_batch, background=None, search_range=5, min_logerr=2)
    with pytest.raises(MimosaError, match="calibration group"):
        check_null_compatibility(stored, sequences=null_batch, background=None, search_range=5,
                                 min_logerr=2, models=[read_model("examples/myog.ihbcp", order=4)])
    with pytest.raises(ValueError, match="one calibration group"):
        build_null([bamm_models[0], read_model("examples/gata2.ihbcp", order=4)], **options)
    variants = [*bamm_models, *[relabel_model(m, seed=s) for m, s in zip(bamm_models, a.contract["model_seeds"])]]
    prepared = [prepare_profile(m, null_batch, min_logerr=2, circular=True) for m in variants]
    for first, second, value in stored.pairs[:8]:
        def index(label):
            role, number, _ = label.split(":", 2)
            return int(number)+(2 if role == "relabeled" else 0)
        assert float(compare(prepared[index(first)], prepared[index(second)]).score) == value


def test_generic_callable_null_equals_native(null_batch):
    models = [read_model(f"examples/{name}.meme") for name in ("foxa2", "gata2")]
    adapted = [WindowModel(m.name, m.motif_length, window_function(m), fingerprint=model_fingerprint(m),
                           calibration_group="test-pwm-v1") for m in models]
    options = dict(sequences=null_batch, n_samples=20, seed=112, min_logerr=2, randomization="input-relabel")
    a, b = build_null(models, **options), build_null(adapted, **options)
    np.testing.assert_array_equal(a.raw_scores, b.raw_scores)
    assert a.pairs == b.pairs


def test_custom_null_requires_explicit_group_and_fingerprint(null_batch):
    a = WindowModel("a", 3, lambda x: x.sum(axis=1))
    b = WindowModel("b", 3, lambda x: x.sum(axis=1))
    with pytest.raises(ValueError, match="calibration_group"):
        build_null([a, b], sequences=null_batch, randomization="input-relabel")
    a.calibration_group = b.calibration_group = "toy"
    with pytest.raises(ValueError, match="fingerprint"):
        build_null([a, b], sequences=null_batch, randomization="input-relabel")


def test_annotation_requires_and_checks_models(null_batch, bamm_models):
    dist = build_null(bamm_models, sequences=null_batch, n_samples=3, min_logerr=2, randomization="input-relabel")
    result = compare(*bamm_models, null_batch, min_logerr=2)
    with pytest.raises(MimosaError, match="compared models"):
        annotate_results([result], dist)
    with pytest.raises(MimosaError, match="calibration group"):
        annotate_results([result], dist, models=[read_model("examples/myog.ihbcp", order=4)])
    assert annotate_results([result], dist, models=bamm_models)[0].p_value is not None


def test_relabeled_profile_cache_reuses_normalization(null_batch, tmp_path):
    source = read_model("examples/myog.ihbcp", order=4)
    model = InputRelabeledModel(source, alphabet_permutations(source.motif_length, 5))
    cache = Cache(tmp_path/"cache")
    a = prepare_profile(model, null_batch, min_logerr=2, circular=True, cache=cache)
    b = prepare_profile(model, null_batch, min_logerr=1, circular=True, cache=cache)
    np.testing.assert_array_equal(a.bundle.forward.data, b.bundle.forward.data)
    assert a.anchors[0].positions.size < b.anchors[0].positions.size


def test_contract_rejects_incomplete_randomization(null_batch, bamm_models):
    dist = build_null(bamm_models, sequences=null_batch, n_samples=2, min_logerr=2, randomization="input-relabel")
    contract = dict(dist.contract)
    contract.pop("model_seeds")
    with pytest.raises(ValueError, match="model_seeds"):
        replace(dist, contract=contract)


def test_cli_bamm_order_null_and_mismatched_order(tmp_path):
    collection = tmp_path/"motifs"
    collection.mkdir()
    for name in ("myog", "gata2"):
        (collection/f"{name}.ihbcp").write_bytes(Path(f"examples/{name}.ihbcp").read_bytes())
    null = tmp_path/"null"
    base = [sys.executable, "-m", "mimosa.cli"]
    result = subprocess.run([*base, "build-null", str(collection), "--output", str(null),
                             "--model-type", "bamm", "--order", "0", "--randomization", "input-relabel",
                             "--num-samples", "4", "--fasta", "examples/foreground.fa", "--min-logerr", "2"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_group"] == "BaMM:order=0"
    compare_args = [*base, "compare", "examples/myog.ihbcp", "examples/gata2.ihbcp",
                    "--query-type", "bamm", "--target-type", "bamm", "--fasta", "examples/foreground.fa",
                    "--min-logerr", "2", "--pvalue", "--null-distribution", str(null)]
    matched = subprocess.run([*compare_args, "--query-order", "0", "--target-order", "0"], capture_output=True, text=True)
    assert matched.returncode == 0, matched.stderr
    assert "p-value" in json.loads(matched.stdout)
    wrong = subprocess.run([*compare_args, "--query-order", "4", "--target-order", "4"], capture_output=True, text=True)
    assert wrong.returncode == 2 and "calibration group" in wrong.stderr
    assert not wrong.stdout


def test_declared_function_fingerprint_changes_with_configuration():
    identity = hashlib.sha256(b"weights-v1").hexdigest()
    a = WindowModel("m", 3, lambda x: x.sum(axis=1), fingerprint=identity, calibration_group="toy-v1")
    b = WindowModel("m", 4, lambda x: x.sum(axis=1), fingerprint=identity, calibration_group="toy-v1")
    assert model_fingerprint(a) != model_fingerprint(b)


def test_missing_randomization_metadata_rejected_on_read(null_batch, bamm_models, tmp_path):
    dist = build_null(bamm_models, sequences=null_batch, n_samples=2, min_logerr=2, randomization="input-relabel")
    path = tmp_path/"null"
    write_null_bundle(path, dist)
    manifest = path/"manifest.toml"
    text = manifest.read_text()
    start = text.index("[randomization]")
    end = text.find("\n[", start+1)
    manifest.write_text(text[:start]+(text[end:] if end >= 0 else ""))
    with pytest.raises(ModelFormatError, match="randomization"):
        read_null_bundle(path)
