import json
import subprocess
import sys

import numpy as np
import pytest

from mimosa import EncodedSequences, PWM, compare, direct_mc, read_model
from mimosa.errors import ModelFormatError


def test_direct_monte_carlo_public_observed_and_order():
    query = read_model("examples/foxa2.meme")
    target = read_model("examples/gata4.meme")
    sequences = EncodedSequences.from_strings(["ACGTN" * 15, "", "TCGA" * 20])
    background = EncodedSequences.from_strings(["ACGT" * 30])
    options = dict(k=2, seed=10, group="gc", min_logerr=1, search_range=1,
                   background=background)
    scores, counts, pvalues = direct_mc(query, [target, query], sequences, **options)
    assert scores.shape == counts.shape == pvalues.shape == (1, 2)
    np.testing.assert_array_equal(pvalues, (counts + 1) / 3)
    for i, model in enumerate((target, query)):
        with pytest.warns(UserWarning, match="the jaccard score is noisy"):
            result = compare(query, model, sequences, min_logerr=1, search_range=1,
                             background=background)
        assert scores[0, i] == result.score


@pytest.mark.parametrize("group", ["full", "gc"])
@pytest.mark.parametrize("algorithm", ["dense", "sparse"])
def test_monte_carlo_counts_match_public_comparison_reference(group, algorithm):
    from mimosa import prepare_profile, relabel_model
    from mimosa.monte_carlo import _permutations

    query = read_model("examples/foxa2.meme")
    targets = [read_model("examples/gata4.meme"), query]
    rng = np.random.default_rng(318)
    sequences = EncodedSequences.from_rows([
        np.empty(0, dtype=np.uint8), rng.integers(0, 5, 700, dtype=np.uint8),
        rng.integers(0, 4, 321, dtype=np.uint8),
    ])
    background = EncodedSequences.from_rows([rng.integers(0, 4, 1200, dtype=np.uint8)])
    options = dict(k=5, seed=91, group=group, min_logerr=1, search_range=2,
                   background=background, algorithm=algorithm, batch_size=3,
                   comparison_block_size=2)
    observed, counts, pvalues = direct_mc(query, targets, sequences, **options)

    def prepare(model):
        return prepare_profile(model, sequences, background=background, min_logerr=1, circular=True)

    fixed = [prepare(target) for target in targets]
    query_profile = prepare(query)
    expected_scores = np.array([[
        compare(query_profile, target, search_range=2).score for target in fixed
    ]], dtype=np.float32)
    expected_counts = np.zeros((1, len(targets)), dtype=np.int64)
    for replicate in range(options["k"]):
        randomized = prepare(relabel_model(
            query, permutations=_permutations(query, options["seed"], replicate, group),
        ))
        for j, target in enumerate(fixed):
            expected_counts[0, j] += (
                compare(randomized, target, search_range=2).score >= expected_scores[0, j]
            )
    np.testing.assert_array_equal(observed, expected_scores)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(pvalues, (expected_counts + 1) / 6)


@pytest.mark.parametrize("algorithm", ["dense", "sparse"])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_monte_carlo_empty_sequences_and_targets(tmp_path, algorithm, storage):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["", ""])
    options = dict(k=3, seed=91, algorithm=algorithm, storage=storage,
                   temp_dir=tmp_path, prefetch=2, batch_size=2)
    observed, counts, pvalues = direct_mc(model, [model, model], sequences, **options)
    np.testing.assert_array_equal(observed, [[0, 0]])
    np.testing.assert_array_equal(counts, [[3, 3]])
    np.testing.assert_array_equal(pvalues, [[1, 1]])
    for result in direct_mc(model, [], sequences, **options):
        assert result.shape == (1, 0)
    assert list(tmp_path.iterdir()) == []


def test_cli_direct_monte_carlo(tmp_path):
    sequences = tmp_path / "input.fa"
    sequences.write_text(">a\n" + "ACGTN" * 12 + "\n")
    command = [
        sys.executable, "-m", "mimosa.cli", "compare-many",
        "examples/foxa2.meme", "examples/gata4.meme", "examples/foxa2.meme",
        "--query-type", "pwm", "--target-type", "pwm", "--fasta", str(sequences),
        "--min-logerr", "1", "--search-range", "1", "--pvalue",
        "--mc-replicates", "2", "--mc-seed", "7", "--mc-group", "gc",
        "--mc-storage", "disk", "--mc-temp-dir", str(tmp_path), "--mc-batch-size", "1",
        "--mc-algorithm", "sparse", "--mc-prefetch", "2",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    results = json.loads(completed.stdout)
    assert len(results) == 2
    for result in results:
        mc = result["monte_carlo"]
        assert (mc["replicates"], mc["seed"], mc["group"]) == (2, 7, "gc")
        assert mc["randomized"] == "query"
        assert mc["version"] == "mimosa-direct-mc-v2"
        assert mc["algorithm"] == "sparse"
        assert result["p-value"] == (mc["exceedances"] + 1) / 3
    assert "temporary target profiles" in completed.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.fa"]


def test_custom_model_monte_carlo():
    import jax.numpy as jnp

    from mimosa import MotifModel

    class Consensus(MotifModel):
        name = "consensus"
        motif_length = 3
        fingerprint_value = "consensus-test-v1"

        def __init__(self, pattern):
            self.pattern = jnp.asarray(pattern, dtype=jnp.uint8)

        def score_sites(self, windows):
            return jnp.sum(windows == self.pattern, axis=1).astype(jnp.float32)

        def fingerprint(self):
            return self.fingerprint_value

    pwm = read_model("examples/foxa2.meme")
    custom = Consensus(np.array([0, 1, 2], dtype=np.uint8))
    sequences = EncodedSequences.from_strings(["ACGTACGTAA" * 6, "TTTGGG" * 10])
    options = dict(k=3, seed=11, min_logerr=1, search_range=2)
    first_counts = None
    for query, target in ((custom, pwm), (pwm, custom), (custom, custom)):
        observed, counts, pvalues = direct_mc(query, [target], sequences, **options)
        with pytest.warns(UserWarning, match="the jaccard score is noisy"):
            result = compare(query, target, sequences, min_logerr=1, search_range=2)
        assert observed[0, 0] == result.score
        np.testing.assert_array_equal(pvalues, (counts + 1) / 4)
        assert counts[0, 0] <= 3
        if first_counts is None:
            first_counts = counts.copy()
    repeat = direct_mc(custom, [pwm], sequences, **options)
    np.testing.assert_array_equal(repeat[1], first_counts)


@pytest.mark.parametrize("group", ["full", "gc"])
def test_custom_mc_reuses_jit_and_matches_relabel_profiles(monkeypatch, group):
    import jax.numpy as jnp

    from mimosa import InputRelabeledModel, WindowModel
    from mimosa.models import DEFAULT_WINDOW_BATCH_SIZE
    import mimosa.monte_carlo as mc

    traces = []

    def score(windows):
        traces.append(windows.shape[0])
        weights = jnp.array([3, 17, 2, 5, 11, 7], dtype=jnp.float32)
        return jnp.sum(windows * weights, axis=1)

    query = WindowModel("contextual", 3, score, left_context=1, right_context=2,
                        fingerprint="contextual-mc-test")
    target = read_model("examples/gata4.meme")
    sequence = "ACGT" * (DEFAULT_WINDOW_BATCH_SIZE // 4)
    foreground = EncodedSequences.from_strings([sequence, "", "TT", "ACGTN"])
    background = EncodedSequences.from_strings(["", "GANTT", "GCAAT" * 10])
    options = dict(k=4, seed=127, group=group, min_logerr=1, search_range=1,
                   background=background, batch_size=2)
    original_profile = mc._profile
    randomized_profiles = []

    def capture(model, sequences, bg, **kwargs):
        result = original_profile(model, sequences, bg, **kwargs)
        if isinstance(model, InputRelabeledModel):
            randomized_profiles.append(np.asarray(result).view(np.uint32))
        return result

    monkeypatch.setattr(mc, "_profile", capture)
    actual = direct_mc(query, [target], foreground, **options)
    # Each shape can be traced once for the observed scorer and once for the
    # randomized scorer, independent of the number of replicas or strands.
    for size in (DEFAULT_WINDOW_BATCH_SIZE, 7, 55):
        assert 1 <= traces.count(size) <= 2
    assert len(randomized_profiles) == options["k"]
    actual_profiles = randomized_profiles.copy()
    randomized_profiles.clear()
    monkeypatch.setattr(mc, "_compile_relabel_scorer", lambda _: None)
    expected = direct_mc(query, [target], foreground, **options)
    np.testing.assert_array_equal(actual_profiles, randomized_profiles)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_custom_mc_rejects_nonfinite_randomized_scores(monkeypatch):
    import jax.numpy as jnp

    from mimosa import WindowModel
    import mimosa.monte_carlo as mc

    def score(windows):
        return jnp.where(windows[:, 0] == 1, jnp.inf, 0.0)

    query = WindowModel("invalid-randomized", 1, score, fingerprint="invalid-randomized")
    target = read_model("examples/gata4.meme")
    sequences = EncodedSequences.from_strings(["A" * 20])
    table = np.array([[1, 0, 3, 2]], dtype=np.uint8)
    monkeypatch.setattr(mc, "_permutations", lambda *args: table)
    with pytest.raises(ModelFormatError, match="non-finite"):
        direct_mc(query, [target], sequences, k=1, seed=127, min_logerr=1)


def test_monte_carlo_rejects_non_motif_inputs():
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    with pytest.raises(TypeError, match="one query MotifModel"):
        direct_mc("model", ["model"], sequences, k=2, seed=1)


def test_removed_null_command_and_option():
    command = [sys.executable, "-m", "mimosa.cli"]
    assert subprocess.run([*command, "build-null"], capture_output=True).returncode == 2
    result = subprocess.run([
        *command, "compare", "examples/foxa2.meme", "examples/gata4.meme",
        "--query-type", "pwm", "--target-type", "pwm",
        "--null-distribution", "old", "--pvalue",
    ], capture_output=True)
    assert result.returncode == 2


@pytest.mark.parametrize("options", [
    {"k": 0, "seed": 1},
    {"k": 2, "seed": -1},
    {"k": 2, "seed": 1, "group": "invalid"},
])
def test_invalid_monte_carlo_options(options):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    with pytest.raises(ValueError):
        direct_mc(model, [model], sequences, **options)


def _overflow_model():
    return PWM("overflow", np.full((5, 4), np.float32(2e38)), (0.25,) * 4)


@pytest.mark.parametrize("bad_side", ["query", "target"])
def test_monte_carlo_rejects_nonfinite_observed_scores(bad_side):
    good = read_model("examples/foxa2.meme")
    bad = _overflow_model()
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    query, target = (bad, good) if bad_side == "query" else (good, bad)
    with pytest.raises(ModelFormatError, match="scan produced non-finite Float32 scores"):
        direct_mc(query, [target], sequences, k=2, seed=1, min_logerr=1)


def test_monte_carlo_rejects_nonfinite_background_scores():
    weights = np.zeros((5, 4), dtype=np.float32)
    weights[[0, 3]] = np.float32(2e38)
    model = PWM("background-overflow", weights, (0.25,) * 4)
    sequences = EncodedSequences.from_strings(["C" * 20])
    background = EncodedSequences.from_strings(["A" * 20])
    with pytest.raises(ModelFormatError, match="scan produced non-finite Float32 scores"):
        direct_mc(model, [model], sequences, background=background, k=2, seed=1, min_logerr=1)


def test_monte_carlo_rejects_nonfinite_randomized_scores(monkeypatch):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    monkeypatch.setattr("mimosa.monte_carlo.relabel_model", lambda *args, **kwargs: _overflow_model())
    with pytest.raises(ModelFormatError, match="scan produced non-finite Float32 scores"):
        direct_mc(model, [model], sequences, k=2, seed=1, min_logerr=1)


@pytest.mark.parametrize("group", ["full", "gc"])
def test_batch_storage_and_target_order_do_not_change_results(tmp_path, group):
    query = read_model("examples/foxa2.meme")
    target = read_model("examples/gata4.meme")
    other = read_model("examples/gata2.meme")
    sequences = EncodedSequences.from_strings(["ACGTN" * 24, "", "TCGA" * 30])
    options = dict(k=5, seed=31, min_logerr=1, search_range=1, group=group)
    expected = direct_mc(query, [target], sequences, batch_size=1, storage="memory", **options)
    actual = direct_mc(query, [other, target, target], sequences, batch_size=3,
                       comparison_block_size=2, storage="disk", temp_dir=tmp_path, **options)
    for first, many in zip(expected, actual):
        assert first[0, 0] == many[0, 1] == many[0, 2]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_comparison_block_size_preserves_counts(block_size):
    query = read_model("examples/foxa2.meme")
    target = read_model("examples/gata4.meme")
    sequences = EncodedSequences.from_strings(["ACGTN" * 30])
    options = dict(k=5, seed=31, min_logerr=1, search_range=1, batch_size=5)
    expected = direct_mc(query, [target], sequences, comparison_block_size=1, **options)
    actual = direct_mc(query, [target], sequences, comparison_block_size=block_size, **options)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_query_replicates_are_prepared_once_and_targets_fixed(monkeypatch):
    import mimosa.monte_carlo as mc

    query = read_model("examples/foxa2.meme")
    targets = [read_model("examples/gata4.meme"), read_model("examples/gata2.meme")]
    sequences = EncodedSequences.from_strings(["ACGT" * 30])
    real_relabel = mc.relabel_model
    seen = []

    def record(model, **kwargs):
        seen.append(model)
        return real_relabel(model, **kwargs)

    monkeypatch.setattr(mc, "relabel_model", record)
    direct_mc(query, targets, sequences, k=3, seed=1, min_logerr=1, batch_size=2)
    assert len(seen) == 3
    assert all(model is query for model in seen)


def test_disk_cleanup_after_error(tmp_path, monkeypatch):
    import mimosa.monte_carlo as mc

    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 30])
    monkeypatch.setattr(mc, "relabel_model", lambda *a, **kw: _overflow_model())
    with pytest.raises(ModelFormatError):
        direct_mc(model, [model], sequences, k=2, seed=1, min_logerr=1,
                  storage="disk", temp_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_disk_capacity_failure_cleans_up(tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("mimosa.monte_carlo.shutil.disk_usage", lambda _: usage(10, 10, 0))
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 30])
    with pytest.raises(OSError, match="temporary space"):
        direct_mc(model, [model], sequences, k=2, seed=1, storage="disk", temp_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_rejects_multiple_queries():
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 20])
    with pytest.raises(TypeError, match="one query"):
        direct_mc([model, model], [model], sequences, k=1, seed=1)


def test_cli_mc_reuses_observed_profiles(tmp_path, monkeypatch, capsys):
    import mimosa.monte_carlo as mc
    from mimosa.cli import main

    path = tmp_path / "sequences.fa"
    path.write_text(">a\n" + "ACGT" * 50 + "\n")
    actual_profile = mc._profile
    calls = []

    def counted(model, sequences, background):
        calls.append(model.name)
        return actual_profile(model, sequences, background)

    monkeypatch.setattr(mc, "_profile", counted)
    args = ["compare-many", "examples/foxa2.meme", "examples/gata2.meme",
            "examples/gata4.meme", "--query-type", "pwm", "--target-type", "pwm",
            "--fasta", str(path), "--min-logerr", "1", "--search-range", "1"]
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main([*args, "--pvalue", "--mc-replicates", "3", "--mc-batch-size", "2"]) == 0
    annotated = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 + 2 + 3
    calls.clear()
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main(args) == 0
    baseline = json.loads(capsys.readouterr().out)
    for observed, result in zip(annotated, baseline):
        for key, value in result.items():
            assert observed[key] == value


@pytest.mark.parametrize("with_background", [False, True])
def test_cli_mc_reuses_persistent_cache(tmp_path, monkeypatch, capsys, with_background):
    import mimosa.monte_carlo as mc
    from mimosa.cli import main

    path = tmp_path / "sequences.fa"
    path.write_text(">a\n" + "ACGTN" * 50 + "\n>b\nTTGCAACG\n")
    background = tmp_path / "background.fa"
    background.write_text(">b\n" + "TCGAA" * 50 + "\n")
    cache = tmp_path / "cache"
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    args = ["compare-many", "examples/foxa2.meme", "examples/gata2.meme",
            "examples/gata4.meme", "--query-type", "pwm", "--target-type", "pwm",
            "--fasta", str(path), "--min-logerr", "1", "--search-range", "1",
            "--pvalue", "--mc-replicates", "3", "--mc-batch-size", "2",
            "--mc-storage", "disk", "--mc-temp-dir", str(temporary), "--mc-prefetch", "2"]
    if with_background:
        args += ["--background", str(background)]
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main(args) == 0
    expected = json.loads(capsys.readouterr().out)
    real_profile = mc._profile
    calls = []

    def counted(model, sequences, bg):
        calls.append(model.name)
        return real_profile(model, sequences, bg)

    monkeypatch.setattr(mc, "_profile", counted)
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main([*args, "--cache-dir", str(cache)]) == 0
    cold = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 + 2 + 3
    assert cold == expected
    calls.clear()
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main([*args, "--cache-dir", str(cache)]) == 0
    warm = json.loads(capsys.readouterr().out)
    assert len(calls) == 3
    assert warm == cold
    assert list(temporary.iterdir()) == []
    assert list(cache.iterdir())
    # Entries written by the MC path remain readable by the regular API.
    from mimosa import prepare_profile, read_fasta
    from mimosa.cache import Cache

    sequences, _ = read_fasta(path)
    bg = read_fasta(background)[0] if with_background else None

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("regular preparation missed the persistent cache")

    import importlib
    monkeypatch.setattr(importlib.import_module("mimosa.scan"), "_scan_batch", unexpected_scan)
    prepare_profile(read_model("examples/foxa2.meme"), sequences, background=bg,
                    min_logerr=1, cache=Cache(str(cache)), circular=True)


def test_cli_cache_compare_equals_compare_many_and_recovers_corruption(tmp_path, monkeypatch, capsys):
    import mimosa.monte_carlo as mc
    from mimosa import read_fasta
    from mimosa.cache import prepared_profile_cache_key
    from mimosa.cli import main

    sequences_path = tmp_path / "sequences.fa"
    sequences_path.write_text(">a\n" + "ACGTN" * 50 + "\n")
    cache = tmp_path / "cache"
    args = ["examples/foxa2.meme", "examples/gata4.meme", "--query-type", "pwm",
            "--target-type", "pwm", "--fasta", str(sequences_path), "--min-logerr", "1",
            "--search-range", "1", "--pvalue", "--mc-replicates", "2",
            "--cache-dir", str(cache)]
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main(["compare", *args]) == 0
    single = json.loads(capsys.readouterr().out)
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main(["compare-many", *args]) == 0
    many = json.loads(capsys.readouterr().out)
    assert many == [single]
    sequences, _ = read_fasta(sequences_path)
    key = prepared_profile_cache_key(read_model("examples/foxa2.meme"), sequences, circular=True)
    data = cache / key / "data.bin"
    with data.open("r+b") as file:
        file.seek(-1, 2)
        value = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([value[0] ^ 1]))
    real = mc._profile
    calls = []

    def counted(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(mc, "_profile", counted)
    with pytest.warns(UserWarning, match="the jaccard score is noisy"):
        assert main(["compare", *args]) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored == single
    assert len(calls) == 3  # One corrupted observed profile plus two random replicates.


@pytest.mark.parametrize("storage", ["memory", "disk"])
@pytest.mark.parametrize("prefetch", [0, 1, 4])
def test_mc_prefetch_preserves_order_and_values(tmp_path, storage, prefetch):
    query = read_model("examples/foxa2.meme")
    targets = [read_model("examples/gata2.meme"), read_model("examples/gata4.meme"), query]
    sequences = EncodedSequences.from_strings(["ACGTN" * 40, "TGCAT" * 30])
    options = dict(k=3, seed=11, min_logerr=2, search_range=1, batch_size=2,
                   storage=storage, temp_dir=tmp_path)
    baseline = direct_mc(query, targets, sequences, prefetch=0, **options)
    actual = direct_mc(query, targets, sequences, prefetch=prefetch, **options)
    for a, b in zip(actual, baseline):
        np.testing.assert_array_equal(a, b)
    assert list(tmp_path.iterdir()) == []


def test_prefetch_cleanup_after_comparison_failure(tmp_path, monkeypatch):
    query = read_model("examples/foxa2.meme")
    target = read_model("examples/gata4.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 30])

    def fail(*args, **kwargs):
        raise RuntimeError("comparison failed")

    monkeypatch.setattr("mimosa.monte_carlo._count_exceedances", fail)
    with pytest.raises(RuntimeError, match="comparison failed"):
        direct_mc(query, [target, query], sequences, k=1, seed=1, min_logerr=1,
                  storage="disk", temp_dir=tmp_path, prefetch=2, algorithm="dense")
    assert list(tmp_path.iterdir()) == []


def test_prefetch_read_failure_cleans_temporary_files(tmp_path, monkeypatch):
    import mimosa.monte_carlo as mc

    query = read_model("examples/foxa2.meme")
    target = read_model("examples/gata4.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 30])
    real_load = mc.np.load

    def fail(path, *args, **kwargs):
        if str(path).endswith("1.npy"):
            raise OSError("simulated prefetch read failure")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(mc.np, "load", fail)
    with pytest.raises(OSError, match="simulated prefetch read failure"):
        direct_mc(query, [target, query], sequences, k=1, seed=1, min_logerr=1,
                  storage="disk", temp_dir=tmp_path, prefetch=2)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [-1, 65, True])
def test_prefetch_rejects_invalid_size(value):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    with pytest.raises((TypeError, ValueError), match="prefetch"):
        direct_mc(model, [model], sequences, k=1, seed=1, prefetch=value)


@pytest.mark.parametrize("group", ["full", "gc"])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_sparse_monte_carlo_matches_dense(tmp_path, monkeypatch, group, storage):
    import mimosa.monte_carlo as mc

    query = read_model("examples/foxa2.meme")
    targets = [read_model("examples/gata2.meme"), read_model("examples/gata4.meme")]
    rng = np.random.default_rng(120)
    sequences = EncodedSequences.from_rows([
        rng.integers(0, 5, n, dtype=np.uint8) for n in (13, 0, 600, 400)
    ])
    options = dict(k=7, seed=42, min_logerr=2, search_range=2, batch_size=5,
                   comparison_block_size=3, group=group, storage=storage, temp_dir=tmp_path)
    expected = direct_mc(query, targets, sequences, algorithm="dense", **options)
    real = mc._count_sparse_exceedances
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mc, "_count_sparse_exceedances", counted)
    actual = direct_mc(query, targets, sequences, algorithm="sparse", **options)
    assert calls
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert list(tmp_path.iterdir()) == []


def test_sparse_monte_carlo_ties_and_dense_fallback(monkeypatch):
    from mimosa.profiles.sparse import make_anchors
    import jax.numpy as jnp

    # Dense tracks do not attempt potentially huge sparse buffers.
    assert make_anchors(jnp.ones((2, 64)), 0.5) is None
    query = read_model("examples/foxa2.meme")
    rng = np.random.default_rng(42)
    sequences = EncodedSequences.from_rows([rng.integers(0, 4, 1000, dtype=np.uint8)])
    monkeypatch.setattr(
        "mimosa.monte_carlo._permutations",
        lambda model, *args: np.tile(np.arange(4), (model.motif_length, 1)),
    )
    observed, counts, pvalues = direct_mc(
        query, [query], sequences, k=3, seed=2, min_logerr=2,
        algorithm="sparse", batch_size=3,
    )
    assert observed[0, 0] == 1
    assert counts[0, 0] == 3
    assert pvalues[0, 0] == 1


def test_invalid_mc_algorithm():
    query = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    with pytest.raises(ValueError, match="algorithm"):
        direct_mc(query, [query], sequences, k=1, seed=1, algorithm="other")


def test_sparse_is_the_default_for_api_and_cli(monkeypatch):
    import mimosa.monte_carlo as mc
    from mimosa.cli import build_parser

    query = read_model("examples/foxa2.meme")
    target = read_model("examples/gata4.meme")
    rng = np.random.default_rng(101)
    sequences = EncodedSequences.from_rows([rng.integers(0, 4, 1000, dtype=np.uint8)])
    real = mc._count_sparse_exceedances
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mc, "_count_sparse_exceedances", counted)
    direct_mc(query, [target], sequences, k=2, seed=1, min_logerr=2)
    assert calls
    parser = build_parser()
    args = parser.parse_args(["compare", "q", "t", "--query-type", "pwm", "--target-type", "pwm"])
    assert args.mc_algorithm == "sparse"
    args = parser.parse_args(["compare", "q", "t", "--query-type", "pwm", "--target-type", "pwm",
                              "--mc-algorithm", "dense"])
    assert args.mc_algorithm == "dense"


def test_sparse_anchor_capacity_is_not_fixed():
    import jax.numpy as jnp
    from mimosa.profiles.sparse import make_anchors

    values = np.zeros((2, 100_000), np.float32)
    values[0, :5000] = 4
    values[1, 3000:8000] = 4
    indices, valid = make_anchors(jnp.asarray(values), 3)
    assert indices.shape == (2, 8192)
    for strand in range(2):
        np.testing.assert_array_equal(np.asarray(indices[strand])[np.asarray(valid[strand])],
                                      np.flatnonzero(values[strand] >= 3))


@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_target_anchor_memory_limit_rebuilds_without_changing_results(tmp_path, monkeypatch, storage):
    import mimosa.monte_carlo as mc

    query = read_model("examples/foxa2.meme")
    targets = [read_model("examples/gata4.meme"), read_model("examples/gata2.meme")]
    rng = np.random.default_rng(42)
    sequences = EncodedSequences.from_rows([rng.integers(0, 4, 1000, dtype=np.uint8)])
    options = dict(k=5, seed=19, min_logerr=2, batch_size=2, comparison_block_size=2,
                   storage=storage, temp_dir=tmp_path, prefetch=2)
    baseline = direct_mc(query, targets, sequences, **options)
    real = mc.make_anchors
    calls = []

    def record(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(mc, "make_anchors", record)
    actual = direct_mc(query, targets, sequences, anchor_memory_limit=0, **options)
    for a, b in zip(actual, baseline):
        np.testing.assert_array_equal(a, b)
    assert len(calls) == 2 + 3 + 2 * 3
    assert all(calls)
    assert list(tmp_path.iterdir()) == []


def test_target_anchor_memory_limit_rejects_negative():
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    with pytest.raises(ValueError, match="memory limits"):
        direct_mc(model, [model], sequences, k=1, seed=1, anchor_memory_limit=-1)


@pytest.mark.parametrize("path,order", [
    ("examples/foxa2.meme", None),
    ("examples/myog.ihbcp", 2),
    ("tests/fixtures/stat_dimont-model-1.xml", None),
    ("tests/fixtures/slim/example-model-1.xml", None),
])
@pytest.mark.parametrize("group", ["full", "gc"])
def test_sparse_matches_dense_for_builtin_query_models(path, order, group, monkeypatch):
    import mimosa.monte_carlo as mc

    query = read_model(path, order=order)
    targets = [read_model("examples/gata4.meme"), query]
    rng = np.random.default_rng(521)
    sequences = EncodedSequences.from_rows([
        rng.integers(0, 5, n, dtype=np.uint8) for n in (1, 0, 407, 601)
    ])
    background = EncodedSequences.from_rows([rng.integers(0, 5, 800, dtype=np.uint8)])
    options = dict(k=3, seed=31, min_logerr=2, search_range=2, batch_size=2,
                   comparison_block_size=2, group=group, background=background)
    expected = direct_mc(query, targets, sequences, algorithm="dense", **options)
    real = mc._count_sparse_exceedances
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mc, "_count_sparse_exceedances", counted)
    actual = direct_mc(query, targets, sequences, algorithm="sparse", **options)
    assert calls
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
