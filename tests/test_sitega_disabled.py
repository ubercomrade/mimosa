"""SiteGA stays in the source tree, but all model entry points are closed."""
import subprocess
import sys

import numpy as np
import pytest

from mimosa import EncodedSequences, SiteGA, compare, prepare_profile, read_model, relabel_model, scan
from mimosa.errors import MimosaError
from mimosa.io import model_fingerprint, read_model_bundle, read_sitega, write_model
from mimosa.randomization import model_calibration_group
import mimosa.models as models


def test_sitega_constructor_and_readers_are_disabled():
    with pytest.raises(MimosaError, match="SiteGA is temporarily disabled"):
        SiteGA("disabled", np.zeros((25, 4), dtype=np.float32), 4)
    with pytest.raises(MimosaError, match="SiteGA is temporarily disabled"):
        read_sitega("examples/sitega_stat6.mat")
    for options in ({}, {"format": "sitega"}):
        with pytest.raises(MimosaError, match="SiteGA is temporarily disabled"):
            read_model("examples/sitega_stat6.mat", **options)


def test_existing_sitega_objects_and_bundles_are_rejected(tmp_path, monkeypatch):
    # Create an existing object/bundle using the dormant implementation, then
    # restore the actual disabled state before calling any public entry point.
    monkeypatch.setattr(models, "_SITEGA_ENABLED", True)
    model = SiteGA("old", np.zeros((25, 4), dtype=np.float32), 4)
    path = tmp_path / "old-model"
    write_model(path, model)
    monkeypatch.setattr(models, "_SITEGA_ENABLED", False)
    batch = EncodedSequences.from_strings(["ACGTACGT"])
    operations = [
        lambda: scan(model, batch),
        lambda: model.scan_into(batch[0], np.empty(5, np.float32), np.empty(5, np.float32)),
        lambda: prepare_profile(model, batch, circular=True),
        lambda: compare(model, model, batch),
        lambda: relabel_model(model),
        lambda: model_calibration_group(model),
        lambda: model_fingerprint(model),
        lambda: write_model(tmp_path / "new-model", model),
        lambda: read_model_bundle(path),
        lambda: read_model(path),
    ]
    for operation in operations:
        with pytest.raises(MimosaError, match="SiteGA is temporarily disabled"):
            operation()
    assert path.is_dir()  # Disabling the reader never deletes existing data.


def test_cli_does_not_offer_sitega():
    command = [sys.executable, "-m", "mimosa.cli", "compare", "--help"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "sitega" not in result.stdout.lower()
    result = subprocess.run(
        [sys.executable, "-m", "mimosa.cli", "compare", "a", "b", "--query-type", "sitega"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not result.stdout
    assert "sitega" in result.stderr


def test_dormant_sitega_scanner_preserves_dinucleotide_scoring(monkeypatch):
    monkeypatch.setattr(models, "_SITEGA_ENABLED", True)
    weights = np.arange(100, dtype=np.float32).reshape(25, 4) / 10
    model = SiteGA("test", weights, 4)
    sequence = np.array([0, 1, 4, 3, 2, 0, 1], dtype=np.uint8)
    fwd, rev = np.empty(4, np.float32), np.empty(4, np.float32)
    model.scan_into(sequence, fwd, rev)
    for i in range(4):
        window = sequence[i:i + 4]
        rc = np.array([4 if b == 4 else 3 - b for b in window[::-1]])
        for values, actual in ((window, fwd[i]), (rc, rev[i])):
            expected = np.float32(0)
            for term in range(3):
                expected = np.float32(expected + weights[5 * values[term] + values[term + 1], term])
            assert actual == expected
