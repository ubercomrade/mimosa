"""Integration tests for the public CLI modes of mimosa."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run CLI through current Python env to avoid global PATH contamination."""
    args = cmd[1:] if cmd and cmd[0] == "mimosa" else cmd
    return subprocess.run([sys.executable, "-m", "mimosa.cli", *args], capture_output=True, text=True)


def cli_json(result: subprocess.CompletedProcess) -> dict:
    """Parse one successful CLI JSON response."""
    return json.loads(result.stdout)


def assert_comparison_output(output: dict, *, metric: str, has_sites: bool = False) -> None:
    """Assert common invariants for one comparison JSON payload."""
    assert isinstance(output["query"], str)
    assert isinstance(output["target"], str)
    assert output["metric"] == metric
    assert output["orientation"] in {"++", "+-", "-+", "--"}
    assert isinstance(output["offset"], int)
    assert isinstance(output["score"], int | float)
    assert math.isfinite(float(output["score"]))
    if has_sites:
        assert isinstance(output["n_sites"], int)
        assert output["n_sites"] >= 0


@pytest.fixture
def examples_dir():
    """Path to examples directory."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


def test_profile_comparison_bamm_vs_pwm(examples_dir, temp_dir):
    """Profile mode should compare bamm vs pwm via scanned profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "myog.ihbcp"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "pwm",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="co", has_sites=True)


def test_profile_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Profile mode should compare bamm vs bamm via scanned profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "gata2.ihbcp"),
        str(examples_dir / "gata4.ihbcp"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="co", has_sites=True)


def test_motif_comparison_pwm_vs_pwm(examples_dir, temp_dir):
    """Motif mode should compare pwm vs pwm directly."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "pif4.meme"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--metric",
        "ed",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="ed")


def test_motif_comparison_bamm_vs_bamm(examples_dir, temp_dir):
    """Motif mode should compare bamm vs bamm directly."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "gata2.ihbcp"),
        str(examples_dir / "gata4.ihbcp"),
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
        "--metric",
        "ed",
        "-v",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="ed")


def test_profile_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Profile mode should compare sitega vs pwm via scanned profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_pwm(examples_dir, temp_dir):
    """Motif mode should compare sitega vs pwm directly."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega_gata2.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "ed",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_pwm_pcc(examples_dir, temp_dir):
    """Motif mode should support PFM-based sitega vs pwm comparison."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "pcc",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_profile_comparison_sitega_vs_pwm_second_case(examples_dir, temp_dir):
    """Profile mode should handle a second sitega vs pwm example."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "sitega.mat"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "pwm",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_sitega_1(examples_dir, temp_dir):
    """Motif mode should compare sitega vs sitega in the first scenario."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "sitega",
        "--metric",
        "pcc",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_sitega_2(examples_dir, temp_dir):
    """Motif mode should compare sitega vs sitega in the second scenario."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "sitega",
        "--metric",
        "ed",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_sitega_vs_sitega_3(examples_dir, temp_dir):
    """Motif mode should compare sitega vs sitega in the third scenario."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "sitega_stat6.mat"),
        str(examples_dir / "sitega_stat6.mat"),
        "--model1-type",
        "sitega",
        "--model2-type",
        "sitega",
        "--metric",
        "ed",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_motif_comparison_pwm_vs_sitega(examples_dir, temp_dir):
    """Motif mode should compare pwm vs sitega directly."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "sitega_gata2.mat"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "sitega",
        "--metric",
        "ed",
        "--pfm-mode",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


def test_profile_comparison_basic(examples_dir, temp_dir):
    """Profile mode should compare two precomputed score profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "co",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="co", has_sites=True)


def test_profile_comparison_accepts_dice_metric(examples_dir, temp_dir):
    """Profile mode should expose the Dice metric through CLI."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "dice",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="dice", has_sites=True)


def test_profile_comparison_accepts_dice_rowwise_metric(examples_dir, temp_dir):
    """Profile CLI should expose the window-averaged rowwise Dice metric."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "dice_rowwise",
        "--window-radius",
        "4",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="dice_rowwise", has_sites=True)


def test_profile_comparison_accepts_cosine_metric(examples_dir, temp_dir):
    """Profile CLI should expose the window-averaged cosine metric."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "cosine",
        "--window-radius",
        "4",
        "--realign-window",
        "2",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="cosine", has_sites=True)


def test_profile_comparison_accepts_co_rowwise_metric(examples_dir, temp_dir):
    """Profile CLI should expose the window-averaged rowwise CO metric."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "co_rowwise",
        "--window-radius",
        "4",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="co_rowwise", has_sites=True)


def test_profile_comparison_rejects_removed_l1sim_metric(examples_dir, temp_dir):
    """Profile CLI should reject removed metrics before running a comparison."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "l1sim",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_profile_comparison_with_empirical_logfpr_thresholding(examples_dir, temp_dir):
    """Profile mode should use threshold-selected site windows on empirically normalized profiles."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "gata4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--fasta",
        str(examples_dir / "foreground.fa"),
        "--metric",
        "co",
        "--min-logfpr",
        "2",
        "--window-radius",
        "6",
        "--realign-window",
        "2",
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    assert_comparison_output(cli_json(result), metric="co", has_sites=True)


def test_profile_comparison_accepts_background_argument(examples_dir, temp_dir):
    """Profile CLI should accept explicit background calibration sequences."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "gata2.meme"),
        str(examples_dir / "gata4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--fasta",
        str(examples_dir / "foreground.fa"),
        "--background",
        str(examples_dir / "background.fa"),
    ]

    result = run_cli(cmd)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"


@pytest.mark.parametrize("removed_flag", ["--permutations", "--distortion", "--min-kernel-size", "--max-kernel-size"])
def test_profile_comparison_rejects_removed_null_flags(examples_dir, temp_dir, removed_flag):
    """Profile mode should reject removed Monte Carlo flags in argparse."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        removed_flag,
        "4",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr.lower()


def test_motif_comparison_rejects_removed_null_flags(examples_dir, temp_dir):
    """Motif mode should reject removed permutation flags in argparse."""
    cmd = [
        "mimosa",
        "motif",
        str(examples_dir / "pif4.meme"),
        str(examples_dir / "pif4.meme"),
        "--model1-type",
        "pwm",
        "--model2-type",
        "pwm",
        "--permutations",
        "10",
        "--permute-rows",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr.lower()


def test_pipeline_with_missing_files():
    """Test pipeline behavior with missing input files."""
    cmd = [
        "mimosa",
        "profile",
        "nonexistent1.ihbcp",
        "nonexistent2.ihbcp",
        "--model1-type",
        "bamm",
        "--model2-type",
        "bamm",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with missing files"
    assert "file not found" in result.stderr.lower(), "Should mention missing file"


def test_profile_comparison_rejects_corr_metric(examples_dir, temp_dir):
    """Profile CLI should reject the removed Pearson metric."""
    cmd = [
        "mimosa",
        "profile",
        str(examples_dir / "scores_1.fasta"),
        str(examples_dir / "scores_2.fasta"),
        "--model1-type",
        "scores",
        "--model2-type",
        "scores",
        "--metric",
        "corr",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with unsupported profile metric"
    assert "invalid choice" in result.stderr.lower()


def test_build_null_and_motif_pvalue_annotation(tmp_path):
    """build-null should create a null distribution file usable by motif comparison."""
    motif_dir = tmp_path / "motifs"
    motif_dir.mkdir()
    (motif_dir / "q.pfm").write_text(">q\n0.7 0.1 0.1 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    (motif_dir / "t1.pfm").write_text(">t1\n0.1 0.7 0.1 0.1\n0.7 0.1 0.1 0.1\n", encoding="utf-8")
    (motif_dir / "t2.pfm").write_text(">t2\n0.1 0.1 0.7 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    groups = tmp_path / "groups.tsv"
    groups.write_text("motif\tgroup\nq\tA\nt1\tB\nt2\tB\n", encoding="utf-8")
    null_distribution_file = tmp_path / "motif-pcc.null.joblib"

    build = run_cli(
        [
            "mimosa",
            "build-null",
            str(motif_dir),
            "--model-type",
            "pwm",
            "--pattern",
            "*.pfm",
            "--groups",
            str(groups),
            "--strategy",
            "motif",
            "--metric",
            "pcc",
            "--output",
            str(null_distribution_file),
        ]
    )
    assert build.returncode == 0, f"Command failed with stderr: {build.stderr}"

    result = run_cli(
        [
            "mimosa",
            "motif",
            str(motif_dir / "q.pfm"),
            str(motif_dir / "t1.pfm"),
            "--model1-type",
            "pwm",
            "--model2-type",
            "pwm",
            "--metric",
            "pcc",
            "--pvalue",
            "--null-distribution",
            str(null_distribution_file),
        ]
    )
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    output = cli_json(result)
    assert_comparison_output(output, metric="pcc")
    assert "p-value" in output
    assert "E-value" in output
    assert "q-value" in output


def test_pipeline_with_invalid_mode():
    """Test pipeline behavior with invalid mode."""
    cmd = [
        "mimosa",
        "invalid_mode",
        "file1",
        "file2",
    ]

    result = run_cli(cmd)
    assert result.returncode != 0, "Should fail with invalid mode"


if __name__ == "__main__":
    pytest.main([__file__])
