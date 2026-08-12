import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimosa import prepare_profile, read_fasta, read_model
from mimosa.cli import _validate_null_compatibility
from mimosa.errors import MimosaError
from mimosa.io.bundles import model_collection_fingerprint, read_null_bundle

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
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["query"] == "scores_1"
        assert d["target"] == "scores_2"
        assert d["orientation"] == "++"
        assert d["offset"] == -6
        assert d["metric"] == "co"
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
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["query"] == "MA0047.3"
        assert d["n_sites"] == 199

    def test_missing_type(self):
        r = run_cli("compare", "a", "b")
        assert r.returncode != 0
        assert "error" in r.stderr

    def test_invalid_metric(self):
        r = run_cli(
            "compare",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            "--metric",
            "bogus",
        )
        assert r.returncode != 0

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

    def test_metric_variant(self):
        r = run_cli(
            "compare",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--query-type",
            "scores",
            "--target-type",
            "scores",
            "--metric",
            "cosine",
        )
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["metric"] == "cosine"

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
            ("--null-distribution", "some-null"),
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


class TestBuildNullCommand:
    def test_build_null(self, tmp_path):
        motifs = tmp_path / "motifs"
        motifs.mkdir()
        for f in ["foxa2.meme", "gata2.meme", "gata4.meme"]:
            with open(f"examples/{f}") as src, open(motifs / f, "w") as dst:
                dst.write(src.read())
        out = tmp_path / "null"
        r = run_cli(
            "build-null",
            str(motifs),
            "--output",
            str(out),
            "--num-samples",
            "10",
            "--fasta",
            "examples/foreground.fa",
            "--background",
            "examples/background.fa",
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["n_models"] == 3
        assert d["n_null"] == 10
        assert out.is_dir()
        assert (out / "manifest.toml").is_file()
        compared = run_cli(
            "compare-many",
            "examples/foxa2.meme",
            "examples/gata2.meme",
            "examples/gata4.meme",
            "--query-type",
            "pwm",
            "--target-type",
            "pwm",
            "--fasta",
            "examples/foreground.fa",
            "--background",
            "examples/background.fa",
            "--pvalue",
            "--null-distribution",
            str(out),
        )
        assert compared.returncode == 0, compared.stderr
        assert all("p-value" in item for item in json.loads(compared.stdout))

    def test_build_null_requires_dir(self, tmp_path):
        r = run_cli(
            "build-null",
            "examples/foreground.fa",
            "--output",
            str(tmp_path / "n"),
        )
        assert r.returncode != 0

    def test_build_null_uses_first_motif_from_each_meme_file(self, tmp_path):
        motifs = tmp_path / "motifs"
        motifs.mkdir()
        multi = motifs / "multi.meme"
        multi.write_text(
            (Path("examples/foxa2.meme").read_text(encoding="utf-8"))
            + "\n"
            + Path("examples/gata4.meme").read_text(encoding="utf-8")
        )
        single = motifs / "single.meme"
        single.write_text(Path("examples/gata2.meme").read_text(encoding="utf-8"))
        output = tmp_path / "null"

        r = run_cli(
            "build-null",
            str(motifs),
            "--output",
            str(output),
            "--num-samples",
            "2",
            "--fasta",
            "examples/foreground.fa",
        )

        assert r.returncode == 0, r.stderr
        stored = read_null_bundle(output)
        assert stored["model_collection_fingerprint"] == model_collection_fingerprint(
            [read_model(multi), read_model(single)]
        )

    @pytest.mark.parametrize("option", ("--num-sequences", "--seq-length", "--num-samples"))
    def test_build_null_rejects_non_positive_generation_limits(self, tmp_path, option):
        r = run_cli(
            "build-null",
            "examples",
            "--output",
            str(tmp_path / "null"),
            option,
            "0",
        )
        assert r.returncode == 2
        assert "must be between" in r.stderr


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

    def test_null_compatibility_rejects_alignment_version(self):
        dist = SimpleNamespace(
            strategy="profile",
            metric="co",
            contract={
                "search_range": 10,
                "window_radius": 10,
                "realign_window": 3,
                "min_logerr": 0.0,
                "alignment_version": "old-alignment",
            },
            sequence_fingerprint="none",
            background_fingerprint="none",
            model_type="pwm",
        )
        with pytest.raises(MimosaError, match="alignment version"):
            _validate_null_compatibility(
                dist,
                metric="co",
                sequences=None,
                background=None,
                search_range=10,
                window_radius=10,
                realign_window=3,
                min_logerr=0.0,
                model_types=("pwm", "pwm"),
            )


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
