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
    scores, counts, pvalues = direct_mc([query], [target, query], sequences, **options)
    assert scores.shape == counts.shape == pvalues.shape == (1, 2)
    np.testing.assert_array_equal(pvalues, (counts + 1) / 3)
    for i, model in enumerate((target, query)):
        result = compare(query, model, sequences, min_logerr=1, search_range=1,
                         background=background)
        assert scores[0, i] == result.score


def test_cli_direct_monte_carlo(tmp_path):
    sequences = tmp_path / "input.fa"
    sequences.write_text(">a\n" + "ACGTN" * 12 + "\n")
    command = [
        sys.executable, "-m", "mimosa.cli", "compare-many",
        "examples/foxa2.meme", "examples/gata4.meme", "examples/foxa2.meme",
        "--query-type", "pwm", "--target-type", "pwm", "--fasta", str(sequences),
        "--min-logerr", "1", "--search-range", "1", "--pvalue",
        "--mc-replicates", "2", "--mc-seed", "7", "--mc-group", "gc",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    results = json.loads(completed.stdout)
    assert len(results) == 2
    for result in results:
        mc = result["monte_carlo"]
        assert (mc["replicates"], mc["seed"], mc["group"]) == (2, 7, "gc")
        assert result["p-value"] == (mc["exceedances"] + 1) / 3


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
        direct_mc([model], [model], sequences, **options)


def _overflow_model():
    return PWM("overflow", np.full((5, 4), np.float32(2e38)), (0.25,) * 4)


@pytest.mark.parametrize("bad_side", ["query", "target"])
def test_monte_carlo_rejects_nonfinite_observed_scores(bad_side):
    good = read_model("examples/foxa2.meme")
    bad = _overflow_model()
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    query, target = (bad, good) if bad_side == "query" else (good, bad)
    with pytest.raises(ModelFormatError, match="scan produced non-finite Float32 scores"):
        direct_mc([query], [target], sequences, k=2, seed=1, min_logerr=1)


def test_monte_carlo_rejects_nonfinite_background_scores():
    weights = np.zeros((5, 4), dtype=np.float32)
    weights[[0, 3]] = np.float32(2e38)
    model = PWM("background-overflow", weights, (0.25,) * 4)
    sequences = EncodedSequences.from_strings(["C" * 20])
    background = EncodedSequences.from_strings(["A" * 20])
    with pytest.raises(ModelFormatError, match="scan produced non-finite Float32 scores"):
        direct_mc([model], [model], sequences, background=background, k=2, seed=1, min_logerr=1)


def test_monte_carlo_rejects_nonfinite_randomized_scores(monkeypatch):
    model = read_model("examples/foxa2.meme")
    sequences = EncodedSequences.from_strings(["ACGT" * 10])
    monkeypatch.setattr("mimosa.monte_carlo.relabel_model", lambda *args, **kwargs: _overflow_model())
    with pytest.raises(ModelFormatError, match="scan produced non-finite Float32 scores"):
        direct_mc([model], [model], sequences, k=2, seed=1, min_logerr=1)
