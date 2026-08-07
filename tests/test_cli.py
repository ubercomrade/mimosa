import json
import os
import subprocess
import sys

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


class TestProfileCommand:
    def test_score_profiles(self):
        r = run_cli(
            "profile",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--model1-type",
            "scores",
            "--model2-type",
            "scores",
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["query"] == "scores_1"
        assert d["target"] == "scores_2"
        assert d["orientation"] == "++"
        assert d["offset"] == -6
        assert d["metric"] == "co"

    def test_motif_models_with_fasta(self):
        r = run_cli(
            "profile",
            "examples/foxa2.meme",
            "examples/gata2.meme",
            "--model1-type",
            "pwm",
            "--model2-type",
            "pwm",
            "--fasta",
            "examples/foreground.fa",
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["query"] == "MA0047.3"
        assert d["n_sites"] == 200

    def test_stdout_only_json(self):
        r = run_cli(
            "profile",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--model1-type",
            "scores",
            "--model2-type",
            "scores",
        )
        json.loads(r.stdout)
        assert "error" not in r.stdout

    def test_missing_type(self):
        r = run_cli("profile", "a", "b")
        assert r.returncode != 0
        assert "error" in r.stderr

    def test_invalid_metric(self):
        r = run_cli(
            "profile",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--model1-type",
            "scores",
            "--model2-type",
            "scores",
            "--metric",
            "bogus",
        )
        assert r.returncode != 0

    def test_missing_file(self):
        r = run_cli(
            "profile",
            "nope.fa",
            "examples/scores_2.fasta",
            "--model1-type",
            "scores",
            "--model2-type",
            "scores",
        )
        assert r.returncode == 2

    def test_metric_variant(self):
        r = run_cli(
            "profile",
            "examples/scores_1.fasta",
            "examples/scores_2.fasta",
            "--model1-type",
            "scores",
            "--model2-type",
            "scores",
            "--metric",
            "cosine",
        )
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["metric"] == "cosine"


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
            "--model-type",
            "pwm",
            "--output",
            str(out),
            "--num-samples",
            "10",
        )
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["n_models"] == 3
        assert d["n_null"] == 10
        assert os.path.isdir(str(out))
        assert os.path.isfile(os.path.join(str(out), "manifest.toml"))

    def test_build_null_requires_dir(self, tmp_path):
        r = run_cli(
            "build-null",
            "examples/foreground.fa",
            "--model-type",
            "pwm",
            "--output",
            str(tmp_path / "n"),
        )
        assert r.returncode != 0

    def test_build_null_requires_pwm(self, tmp_path):
        r = run_cli(
            "build-null",
            "examples",
            "--model-type",
            "sitega",
            "--output",
            str(tmp_path / "n"),
        )
        assert r.returncode != 0


class TestCacheCommand:
    def test_clear(self, tmp_path):
        cache_dir = tmp_path / "cache"
        from mimosa.cache import Cache, cache_set

        cache_set(Cache(str(cache_dir)), "abc", b"x")
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

    def test_no_command(self):
        r = run_cli()
        assert r.returncode != 0
