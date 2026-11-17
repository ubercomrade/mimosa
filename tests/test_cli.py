import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mimosa import prepare_profile, read_fasta, read_model
from mimosa.errors import MimosaError

MIMOSA = [sys.executable, "-m", "mimosa.cli"]


def run_cli(*args, **kwargs):
    result = subprocess.run(
        [*MIMOSA, *args],
        capture_output=True,
        text=True,
        cwd=kwargs.pop("cwd", None),
        **kwargs,
    )
    return result


class TestCompareCommand:
    def test_score_profiles(self):
        r = run_cli(
            "compare",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            "--min-logerr",
            "2",
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["query"] == "scores_1"
        assert d["target"] == "scores_2"
        assert d["orientation"] == "++"
        assert d["offset"] == 4
        assert d["metric"] == "jaccard"
        assert "error" not in r.stdout

    def test_motif_models_with_fasta(self):
        r = run_cli(
            "compare",
            "examples/foxa2.meme",
            "examples/gata2.meme",
            "--query-type",
            "pwm",
            "--target-type",
            "pwm",
            "--fasta",
            "examples/foreground.fa",
            "--min-logerr",
            "2",
        )
        assert r.returncode == 0, r.stderr
        assert r.stderr == ""
        d = json.loads(r.stdout)
        assert d["query"] == "MA0047.3"
        assert d["n_sites"] == 2828

    def test_defaults_use_generated_sequences_and_threshold_four(self):
        r = run_cli(
            "compare",
            "examples/foxa2.meme",
            "examples/foxa2.meme",
            "--query-type",
            "pwm",
            "--target-type",
            "pwm",
        )
        assert r.returncode == 0, r.stderr
        assert r.stderr == ""
        d = json.loads(r.stdout)
        assert d["score"] == 1.0
        assert d["offset"] == 0
        assert 50 < d["n_sites"] < 200

    def test_few_positions_warn_on_stderr_only(self):
        r = run_cli(
            "compare",
            "examples/foxa2.meme",
            "examples/gata2.meme",
            "--query-type",
            "pwm",
            "--target-type",
            "pwm",
            "--fasta",
            "examples/foreground.fa",
        )
        assert r.returncode == 0, r.stderr
        assert "noisy" in r.stderr
        assert json.loads(r.stdout)["metric"] == "jaccard"

    def test_missing_type(self):
        r = run_cli("compare", "a", "b")
        assert r.returncode != 0
        assert "error" in r.stderr

    @pytest.mark.parametrize(
        "option", ("--metric", "--window-radius", "--circular")
    )
    def test_removed_options_are_rejected(self, option):
        r = run_cli(
            "compare",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            option,
            "1",
        )
        assert r.returncode == 2
        assert "unrecognized arguments" in r.stderr

    def test_missing_file(self):
        r = run_cli(
            "compare",
            "nope.fa",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
        )
        assert r.returncode == 2

    @pytest.mark.parametrize("option", ("--fasta", "--background"))
    def test_scores_only_rejects_unused_sequence_options(self, option):
        r = run_cli(
            "compare",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            option,
            "examples/foreground.fa",
        )
        assert r.returncode == 2
        assert "scores-only" in r.stderr

    @pytest.mark.parametrize(
        "option",
        (
            ("--effective-number-of-targets", "2"),
        ),
    )
    def test_significance_options_require_pvalue(self, option):
        r = run_cli(
            "compare",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            *option,
        )
        assert r.returncode == 2
        assert "requires --pvalue" in r.stderr


class TestCompareManyCommand:
    def test_json_array_preserves_target_order_and_type(self):
        r = run_cli(
            "compare-many",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "examples/scores_1.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
        )
        assert r.returncode == 0, r.stderr
        results = json.loads(r.stdout)
        assert isinstance(results, list)
        assert [result["target"] for result in results] == ["scores_2", "scores_1"]

    @pytest.mark.parametrize(
        "option",
        (("--total-threads", "3", "--numba-threads", "2"), ("--numba-threads", "5")),
    )
    def test_rejects_invalid_thread_budget(self, option):
        r = run_cli(
            "compare-many",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            *option,
        )
        assert r.returncode != 0


class TestCacheCommand:
    def test_clear(self, tmp_path):
        cache_dir = tmp_path / "cache"
        from mimosa.cache import Cache

        prepare_profile(
            read_model("examples/foxa2.meme"),
            read_fasta("examples/foreground.fa")[0],
            cache=Cache(str(cache_dir)),
        )
        r = run_cli("cache", "clear", "--cache-dir", str(cache_dir))
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["removed"] == 1

    def test_clear_missing_dir(self, tmp_path):
        r = run_cli("cache", "clear", "--cache-dir", str(tmp_path / "none"))
        assert r.returncode == 0
        assert json.loads(r.stdout)["removed"] == 0

    def test_clear_broad_dir(self):
        r = run_cli("cache", "clear", "--cache-dir", "/")
        assert r.returncode != 0

class TestVersion:
    def test_version(self):
        r = run_cli("--version")
        assert r.returncode == 0
        assert "mimosa" in r.stdout

    def test_import_does_not_rewrite_numba_environment_from_host_argv(self):
        code = (
            "import os, sys; "
            "sys.argv = ['host', 'compare', '--numba-threads=2']; "
            "import mimosa; "
            "print(os.environ['NUMBA_NUM_THREADS'])"
        )
        env = {**os.environ, "NUMBA_NUM_THREADS": "7"}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "7"

    def test_no_command(self):
        r = run_cli()
        assert r.returncode != 0


def test_typed_read_autodetects_saved_context_bundle(tmp_path):
    from mimosa import BaMM
    from mimosa.cli import _read_typed_model
    from mimosa.io.bundles import write_model

    model = BaMM("b", np.zeros((25, 4), dtype=np.float32), 1, 4)
    write_model(tmp_path / "bundle", model)
    assert isinstance(_read_typed_model(tmp_path / "bundle", "bamm"), BaMM)
    with pytest.raises(MimosaError):
        _read_typed_model(tmp_path / "bundle", "pwm")
