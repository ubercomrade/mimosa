"""Generate frozen oracle fixtures for Julia compatibility testing.

This script is pinned to a specific Python commit and environment.
It produces deterministic intermediate and final values that Julia tests
load from disk without running Python at test time.

Usage:
    uv run python scripts/generate_oracle_fixtures.py

Output:
    tests/fixtures/compatibility/manifest.json
    tests/fixtures/compatibility/*.npy
    tests/fixtures/compatibility/*.json

Regenerate only after reviewing the diff and updating the manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from mimosa.batches import make_random_sequence_batch
from mimosa.comparison import compare, create_comparator_config
from mimosa.functions import pfm_to_pwm
from mimosa.functions.tails import build_score_log_tail_table
from mimosa.io import read_fasta, read_meme, read_pfm, read_scores
from mimosa.models import read_model
from mimosa.scanning import (
    flatten_scan_scores,
    scan_model,
    scan_model_strands,
    score_bounds_from_model,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "compatibility"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _save_npy(name: str, arr: np.ndarray) -> dict:
    path = FIXTURE_DIR / f"{name}.npy"
    np.save(path, arr)
    return {"file": path.name, "dtype": str(arr.dtype), "shape": list(arr.shape), "checksum": _sha256_array(arr)}


def _save_json(name: str, data: dict) -> dict:
    path = FIXTURE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return {"file": path.name, "checksum": _sha256_file(path)}


def _ensure_dir() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_pwm_parsing_fixtures() -> list[dict]:
    """Fixtures for MEME and PFM parsing."""
    fixtures = []

    # MEME parsing: examples/pif4.meme, index 0
    pfm, info, count = read_meme(str(EXAMPLES_DIR / "pif4.meme"), index=0)
    name, length = info
    fixtures.append(
        {
            "id": "pwm_parse_meme_pif4",
            "description": "Parse pif4.meme index 0 to PFM array",
            "arrays": {"pfm": _save_npy("pwm_parse_meme_pif4__pfm", pfm)},
            "metadata": {"name": name, "length": int(length), "motif_count": int(count), "shape": list(pfm.shape)},
        }
    )

    # PFM file parsing: examples/pif4.pfm
    pfm2, length2 = read_pfm(str(EXAMPLES_DIR / "pif4.pfm"))
    fixtures.append(
        {
            "id": "pwm_parse_pfm_pif4",
            "description": "Parse pif4.pfm to PFM array",
            "arrays": {"pfm": _save_npy("pwm_parse_pfm_pif4__pfm", pfm2)},
            "metadata": {"length": int(length2), "shape": list(pfm2.shape)},
        }
    )

    # PFM to PWM conversion
    pwm = pfm_to_pwm(pfm)
    fixtures.append(
        {
            "id": "pwm_to_pwm_from_pif4",
            "description": "Convert pif4 PFM to PWM via pfm_to_pwm",
            "arrays": {"pwm": _save_npy("pwm_to_pwm_from_pif4__pwm", pwm)},
            "metadata": {"shape": list(pwm.shape)},
        }
    )

    return fixtures


def generate_sequence_fixtures() -> list[dict]:
    """Fixtures for FASTA reading and encoding."""
    fixtures = []

    # FASTA reading: examples/foreground.fa
    batch = read_fasta(str(EXAMPLES_DIR / "foreground.fa"))
    fixtures.append(
        {
            "id": "fasta_read_foreground",
            "description": "Read foreground.fa to encoded sequence batch",
            "arrays": {
                "values": _save_npy("fasta_read_foreground__values", batch["values"]),
                "lengths": _save_npy("fasta_read_foreground__lengths", batch["lengths"]),
            },
            "metadata": {
                "n_sequences": int(batch["lengths"].shape[0]),
                "max_length": int(batch["values"].shape[1]),
                "padding_value": int(batch["padding_value"]),
            },
        }
    )

    # Random sequence batch (deterministic seed)
    rng_batch = make_random_sequence_batch(10, 50, seed=127)
    fixtures.append(
        {
            "id": "random_sequence_batch_seed127",
            "description": "Random encoded sequence batch, seed=127, n=10, len=50",
            "arrays": {
                "values": _save_npy("random_sequence_batch_seed127__values", rng_batch["values"]),
                "lengths": _save_npy("random_sequence_batch_seed127__lengths", rng_batch["lengths"]),
            },
            "metadata": {
                "n_sequences": 10,
                "seq_length": 50,
                "seed": 127,
                "padding_value": int(rng_batch["padding_value"]),
            },
        }
    )

    return fixtures


def generate_reverse_complement_fixtures() -> list[dict]:
    """Fixtures for reverse complement of PWM and sequences."""
    fixtures = []

    pfm, _info, _ = read_meme(str(EXAMPLES_DIR / "pif4.meme"), index=0)
    pwm = pfm_to_pwm(pfm)

    # Reverse complement of PWM: flip rows (complement) and reverse columns (position)
    pwm_rc = pwm[::-1, ::-1].copy()
    fixtures.append(
        {
            "id": "pwm_reverse_complement_pif4",
            "description": "Reverse complement of pif4 PWM (flip rows + reverse columns)",
            "arrays": {
                "pwm_forward": _save_npy("pwm_reverse_complement_pif4__forward", pwm),
                "pwm_reverse": _save_npy("pwm_reverse_complement_pif4__reverse", pwm_rc),
            },
            "metadata": {"shape": list(pwm.shape)},
        }
    )

    return fixtures


def generate_scan_fixtures() -> list[dict]:
    """Fixtures for PWM scanning on deterministic sequences."""
    fixtures = []

    # Load PWM model
    model = read_model(str(EXAMPLES_DIR / "pif4.meme"), "pwm")

    # Scan on random sequences
    seq_batch = make_random_sequence_batch(50, 200, seed=42)

    # Save input sequences so Julia compatibility tests can load them
    # without needing to replicate NumPy's default_rng.
    fixtures.append(
        {
            "id": "pwm_scan_input_seed42",
            "description": "Input sequences for PWM scan fixtures (seed=42, n=50, len=200)",
            "arrays": {
                "values": _save_npy("pwm_scan_input_seed42__values", seq_batch["values"]),
                "lengths": _save_npy("pwm_scan_input_seed42__lengths", seq_batch["lengths"]),
            },
            "metadata": {
                "n_sequences": 50,
                "seq_length": 200,
                "seed": 42,
                "padding_value": int(seq_batch["padding_value"]),
            },
        }
    )

    # Forward scan
    forward_scores = scan_model(model, seq_batch, strand="+")
    fixtures.append(
        {
            "id": "pwm_scan_forward_pif4_seed42",
            "description": "Forward PWM scan of pif4 on 50 random sequences (seed=42, len=200)",
            "arrays": {
                "values": _save_npy("pwm_scan_forward_pif4_seed42__values", forward_scores["values"]),
                "mask": _save_npy("pwm_scan_forward_pif4_seed42__mask", forward_scores["mask"]),
                "lengths": _save_npy("pwm_scan_forward_pif4_seed42__lengths", forward_scores["lengths"]),
            },
            "metadata": {
                "n_sequences": 50,
                "seq_length": 200,
                "seed": 42,
                "motif_length": int(model.length),
                "padding_value": float(forward_scores["padding_value"]),
            },
        }
    )

    # Reverse scan
    reverse_scores = scan_model(model, seq_batch, strand="-")
    fixtures.append(
        {
            "id": "pwm_scan_reverse_pif4_seed42",
            "description": "Reverse PWM scan of pif4 on 50 random sequences (seed=42, len=200)",
            "arrays": {
                "values": _save_npy("pwm_scan_reverse_pif4_seed42__values", reverse_scores["values"]),
                "mask": _save_npy("pwm_scan_reverse_pif4_seed42__mask", reverse_scores["mask"]),
                "lengths": _save_npy("pwm_scan_reverse_pif4_seed42__lengths", reverse_scores["lengths"]),
            },
            "metadata": {
                "n_sequences": 50,
                "seq_length": 200,
                "seed": 42,
                "motif_length": int(model.length),
                "padding_value": float(reverse_scores["padding_value"]),
            },
        }
    )

    # Both strands
    both_bundle = scan_model_strands(model, seq_batch)
    fixtures.append(
        {
            "id": "pwm_scan_both_pif4_seed42",
            "description": "Both-strand PWM scan bundle of pif4 on 50 random sequences (seed=42, len=200)",
            "arrays": {
                "values": _save_npy("pwm_scan_both_pif4_seed42__values", both_bundle["values"]),
                "lengths": _save_npy("pwm_scan_both_pif4_seed42__lengths", both_bundle["lengths"]),
            },
            "metadata": {
                "n_sequences": 50,
                "seq_length": 200,
                "seed": 42,
                "motif_length": int(model.length),
                "padding_value": float(both_bundle["padding_value"]),
                "bundle_shape": list(both_bundle["values"].shape),
            },
        }
    )

    # Score bounds
    min_score, max_score = score_bounds_from_model(model)
    fixtures.append(
        {
            "id": "pwm_score_bounds_pif4",
            "description": "Theoretical score bounds for pif4 PWM",
            "metadata": {"min_score": float(min_score), "max_score": float(max_score)},
        }
    )

    return fixtures


def generate_normalization_fixtures() -> list[dict]:
    """Fixtures for empirical log-tail normalization."""
    fixtures = []

    model = read_model(str(EXAMPLES_DIR / "pif4.meme"), "pwm")
    seq_batch = make_random_sequence_batch(50, 200, seed=42)

    # Build log-tail table from scan scores
    scores = scan_model(model, seq_batch, strand="both")
    flat_scores = flatten_scan_scores(scores)
    table = build_score_log_tail_table(flat_scores)

    fixtures.append(
        {
            "id": "normalization_log_tail_pif4_seed42",
            "description": "Empirical log-tail table from pif4 both-strand scan scores (seed=42)",
            "arrays": {
                "flat_scores": _save_npy("normalization_log_tail_pif4_seed42__flat_scores", flat_scores),
                "table": _save_npy("normalization_log_tail_pif4_seed42__table", table),
            },
            "metadata": {
                "n_scores": int(flat_scores.shape[0]),
                "table_shape": list(table.shape),
            },
        }
    )

    return fixtures


def generate_motif_alignment_fixtures() -> list[dict]:
    """Fixtures for direct motif matrix comparison."""
    fixtures = []

    # Self-comparison: pif4 vs pif4 (should give score ~1.0 for pcc, offset 0, orientation ++)
    model1 = read_model(str(EXAMPLES_DIR / "pif4.meme"), "pwm")
    model2 = read_model(str(EXAMPLES_DIR / "pif4.meme"), "pwm")

    for metric in ["pcc", "ed", "cosine"]:
        config = create_comparator_config(metric=metric)
        result = compare(model1, model2, "motif", config)
        fixtures.append(
            {
                "id": f"motif_alignment_self_pif4_{metric}",
                "description": f"Direct motif self-comparison pif4 vs pif4, metric={metric}",
                "metadata": {
                    "query": result["query"],
                    "target": result["target"],
                    "score": float(result["score"]),
                    "offset": int(result["offset"]),
                    "orientation": result["orientation"],
                    "metric": result["metric"],
                },
            }
        )

    # Cross-comparison: pif4.meme vs gata2.meme
    model_gata2 = read_model(str(EXAMPLES_DIR / "gata2.meme"), "pwm")
    for metric in ["pcc", "ed", "cosine"]:
        config = create_comparator_config(metric=metric)
        result = compare(model1, model_gata2, "motif", config)
        fixtures.append(
            {
                "id": f"motif_alignment_pif4_vs_gata2_{metric}",
                "description": f"Direct motif comparison pif4 vs gata2, metric={metric}",
                "metadata": {
                    "query": result["query"],
                    "target": result["target"],
                    "score": float(result["score"]),
                    "offset": int(result["offset"]),
                    "orientation": result["orientation"],
                    "metric": result["metric"],
                },
            }
        )

    return fixtures


def generate_score_profile_fixtures() -> list[dict]:
    """Fixtures for score profile reading."""
    fixtures = []

    scores1 = read_scores(str(EXAMPLES_DIR / "scores_1.fasta"))
    fixtures.append(
        {
            "id": "score_profile_read_1",
            "description": "Read scores_1.fasta to masked score batch",
            "arrays": {
                "values": _save_npy("score_profile_read_1__values", scores1["values"]),
                "mask": _save_npy("score_profile_read_1__mask", scores1["mask"]),
                "lengths": _save_npy("score_profile_read_1__lengths", scores1["lengths"]),
            },
            "metadata": {
                "n_profiles": int(scores1["lengths"].shape[0]),
                "max_length": int(scores1["values"].shape[1]),
                "padding_value": float(scores1["padding_value"]),
            },
        }
    )

    scores2 = read_scores(str(EXAMPLES_DIR / "scores_2.fasta"))
    fixtures.append(
        {
            "id": "score_profile_read_2",
            "description": "Read scores_2.fasta to masked score batch",
            "arrays": {
                "values": _save_npy("score_profile_read_2__values", scores2["values"]),
                "mask": _save_npy("score_profile_read_2__mask", scores2["mask"]),
                "lengths": _save_npy("score_profile_read_2__lengths", scores2["lengths"]),
            },
            "metadata": {
                "n_profiles": int(scores2["lengths"].shape[0]),
                "max_length": int(scores2["values"].shape[1]),
                "padding_value": float(scores2["padding_value"]),
            },
        }
    )

    return fixtures


def generate_profile_comparison_fixtures() -> list[dict]:
    """Fixtures for profile comparison with precomputed scores."""
    from mimosa.handlers import register_builtin_handlers

    register_builtin_handlers()
    fixtures = []

    model1 = read_model(str(EXAMPLES_DIR / "scores_1.fasta"), "scores")
    model2 = read_model(str(EXAMPLES_DIR / "scores_2.fasta"), "scores")

    for metric in ["co", "co_rowwise", "dice", "dice_rowwise", "cosine"]:
        config = create_comparator_config(metric=metric, search_range=0, window_radius=0)
        result = compare(model1, model2, "profile", config)
        fixtures.append(
            {
                "id": f"profile_comparison_scores_{metric}_zero_shift",
                "description": (
                    f"Profile comparison of precomputed scores, metric={metric}, search_range=0, window_radius=0"
                ),
                "metadata": {
                    "query": result["query"],
                    "target": result["target"],
                    "score": float(result["score"]),
                    "offset": int(result["offset"]),
                    "orientation": result["orientation"],
                    "metric": result["metric"],
                    "n_sites": int(result["n_sites"]) if result["n_sites"] is not None else None,
                },
            }
        )

    return fixtures


def generate_site_reconstruction_fixtures() -> list[dict]:
    """Fixtures for site extraction and PFM reconstruction."""
    from mimosa.sites import get_pfm, get_sites

    fixtures = []

    model = read_model(str(EXAMPLES_DIR / "pif4.meme"), "pwm")
    seq_batch = make_random_sequence_batch(100, 200, seed=42)

    # Best-mode sites
    sites_df = get_sites(model, seq_batch, mode="best", strand="both")
    fixtures.append(
        {
            "id": "sites_best_pif4_seed42",
            "description": "Best-mode site extraction for pif4 on 100 random sequences (seed=42)",
            "arrays": {
                "seq_index": _save_npy("sites_best_pif4_seed42__seq_index", sites_df["seq_index"].to_numpy()),
                "start": _save_npy("sites_best_pif4_seed42__start", sites_df["start"].to_numpy()),
                "end": _save_npy("sites_best_pif4_seed42__end", sites_df["end"].to_numpy()),
                "score": _save_npy("sites_best_pif4_seed42__score", sites_df["score"].to_numpy()),
                "log_tail": _save_npy("sites_best_pif4_seed42__log_tail", sites_df["log_tail"].to_numpy()),
            },
            "metadata": {
                "n_sites": int(len(sites_df)),
                "motif_length": int(model.length),
                "n_sequences": 100,
                "seed": 42,
            },
        }
    )

    # PFM reconstruction
    pfm = get_pfm(model, seq_batch, mode="best", strand="both", pseudocount=0.25)
    fixtures.append(
        {
            "id": "pfm_reconstruction_best_pif4_seed42",
            "description": "PFM reconstruction from best sites for pif4 (seed=42, pseudocount=0.25)",
            "arrays": {"pfm": _save_npy("pfm_reconstruction_best_pif4_seed42__pfm", pfm)},
            "metadata": {
                "motif_length": int(model.length),
                "n_sequences": 100,
                "seed": 42,
                "pseudocount": 0.25,
                "shape": list(pfm.shape),
            },
        }
    )

    return fixtures


def generate_gev_fixtures() -> list[dict]:
    """Fixtures for GEV fitting compatibility."""
    from scipy import stats

    fixtures = []

    # Sample 1: 200 Gumbel-distributed scores
    rng = np.random.default_rng(12345)
    scores_gumbel = rng.gumbel(loc=0.0, scale=1.0, size=200)
    scores_gumbel = np.sort(scores_gumbel)
    params_gumbel = stats.genextreme.fit(scores_gumbel)
    sf_points_gumbel = np.array([stats.genextreme.sf(s, *params_gumbel) for s in [-1.0, 0.0, 1.0, 2.0, 5.0]])

    fixtures.append(
        {
            "id": "gev_fit_gumbel_200",
            "description": "GEV fit on 200 Gumbel(0,1) scores",
            "arrays": {
                "scores": _save_npy("gev_fit_gumbel_200__scores", scores_gumbel),
                "sf_points": _save_npy("gev_fit_gumbel_200__sf_points", sf_points_gumbel),
            },
            "metadata": {
                "n": 200,
                "genextreme_params": [float(p) for p in params_gumbel],
                "sf_query_points": [-1.0, 0.0, 1.0, 2.0, 5.0],
                "scipy_version": scipy.__version__,
            },
        }
    )

    # Sample 2: 2000 normal scores
    scores_normal = rng.standard_normal(2000)
    scores_normal = np.sort(scores_normal)
    params_normal = stats.genextreme.fit(scores_normal)
    sf_points_normal = np.array([stats.genextreme.sf(s, *params_normal) for s in [-1.0, 0.0, 1.0, 2.0, 5.0]])

    fixtures.append(
        {
            "id": "gev_fit_normal_2000",
            "description": "GEV fit on 2000 Normal(0,1) scores",
            "arrays": {
                "scores": _save_npy("gev_fit_normal_2000__scores", scores_normal),
                "sf_points": _save_npy("gev_fit_normal_2000__sf_points", sf_points_normal),
            },
            "metadata": {
                "n": 2000,
                "genextreme_params": [float(p) for p in params_normal],
                "sf_query_points": [-1.0, 0.0, 1.0, 2.0, 5.0],
                "scipy_version": scipy.__version__,
            },
        }
    )

    # Sample 3: 500 exponential scores
    scores_exp = rng.exponential(1.0, 500)
    scores_exp = np.sort(scores_exp)
    params_exp = stats.genextreme.fit(scores_exp)
    sf_points_exp = np.array([stats.genextreme.sf(s, *params_exp) for s in [0.5, 1.0, 2.0, 5.0, 10.0]])

    fixtures.append(
        {
            "id": "gev_fit_exponential_500",
            "description": "GEV fit on 500 Exp(1) scores",
            "arrays": {
                "scores": _save_npy("gev_fit_exponential_500__scores", scores_exp),
                "sf_points": _save_npy("gev_fit_exponential_500__sf_points", sf_points_exp),
            },
            "metadata": {
                "n": 500,
                "genextreme_params": [float(p) for p in params_exp],
                "sf_query_points": [0.5, 1.0, 2.0, 5.0, 10.0],
                "scipy_version": scipy.__version__,
            },
        }
    )

    # Sample 4: 5000 uniform scores
    scores_uniform = rng.uniform(0, 1, 5000)
    scores_uniform = np.sort(scores_uniform)
    params_uniform = stats.genextreme.fit(scores_uniform)
    sf_points_uniform = np.array([stats.genextreme.sf(s, *params_uniform) for s in [0.2, 0.5, 0.8, 0.95, 1.0]])

    fixtures.append(
        {
            "id": "gev_fit_uniform_5000",
            "description": "GEV fit on 5000 Uniform(0,1) scores",
            "arrays": {
                "scores": _save_npy("gev_fit_uniform_5000__scores", scores_uniform),
                "sf_points": _save_npy("gev_fit_uniform_5000__sf_points", sf_points_uniform),
            },
            "metadata": {
                "n": 5000,
                "genextreme_params": [float(p) for p in params_uniform],
                "sf_query_points": [0.2, 0.5, 0.8, 0.95, 1.0],
                "scipy_version": scipy.__version__,
            },
        }
    )

    return fixtures



def generate_bamm_fixtures() -> list[dict]:
    """Fixtures for BaMM parsing, scanning, and score bounds."""
    from mimosa.io.bamm import parse_file_content, read_bamm

    fixtures = []

    # 1. Parse BaMM and save representation for different files and orders
    for fname in ["myog.ihbcp", "gata2.ihbcp", "foxa2.ihbcp"]:
        path = str(EXAMPLES_DIR / fname)
        raw_data, max_order, n_positions = parse_file_content(path)

        for target_order in [0, 1, min(2, max_order)]:
            rep = read_bamm(path, target_order)
            base = fname.replace(".ihbcp", "")
            fixture_id = f"bamm_parse_{base}_order{target_order}"
            fixtures.append(
                {
                    "id": fixture_id,
                    "description": f"Parse {fname} with target_order={target_order}",
                    "arrays": {"representation": _save_npy(fixture_id + "__representation", rep)},
                    "metadata": {
                        "name": base,
                        "max_order": max_order,
                        "target_order": target_order,
                        "motif_length": n_positions,
                        "shape": list(rep.shape),
                    },
                }
            )

    # 2. Generate random sequences for BaMM scanning (5-ary encoding)
    rng = np.random.default_rng(42)
    n_seq = 50
    seq_len = 200
    values = rng.integers(0, 5, size=(n_seq, seq_len), dtype=np.int8)
    lengths = np.full(n_seq, seq_len, dtype=np.int64)

    seq_fixture_id = "bamm_scan_input_seed42"
    fixtures.append(
        {
            "id": seq_fixture_id,
            "description": "Input sequences for BaMM scan fixtures (seed=42, n=50, len=200, 5-ary encoding)",
            "arrays": {
                "values": _save_npy(seq_fixture_id + "__values", values),
                "lengths": _save_npy(seq_fixture_id + "__lengths", lengths),
            },
            "metadata": {"n_sequences": 50, "seq_length": 200, "seed": 42, "padding_value": 4},
        }
    )

    # 3. BaMM scanning (forward and reverse) using inline kernels
    def _scan_forward(values, lengths, model_rows, kmer, motif_len):
        context_len = kmer - 1
        window_size = motif_len + context_len
        n_terms = window_size - kmer + 1
        n_rows = values.shape[0]
        max_scores = max(values.shape[1] - window_size + 1, 0)
        scores = np.zeros((n_rows, max_scores), dtype=np.float32)
        mask = np.zeros((n_rows, max_scores), dtype=bool)
        for row in range(n_rows):
            length = int(lengths[row])
            n_pos = max(length - window_size + 1, 0)
            if n_pos == 0:
                continue
            for pos in range(n_pos):
                total = np.float32(0.0)
                for term in range(n_terms):
                    code = 0
                    src_start = pos - context_len + term
                    for offset in range(kmer):
                        src = src_start + offset
                        encoded = 4
                        if 0 <= src < length:
                            encoded = int(values[row, src])
                        code = code * 5 + encoded
                    total += model_rows[code, term]
                scores[row, pos] = total
                mask[row, pos] = True
        return scores, mask

    def _scan_reverse(values, lengths, model_rows, kmer, motif_len):
        context_len = kmer - 1
        window_size = motif_len + context_len
        n_terms = window_size - kmer + 1
        n_rows = values.shape[0]
        max_scores = max(values.shape[1] - window_size + 1, 0)
        scores = np.zeros((n_rows, max_scores), dtype=np.float32)
        mask = np.zeros((n_rows, max_scores), dtype=bool)
        for row in range(n_rows):
            length = int(lengths[row])
            n_pos = max(length - window_size + 1, 0)
            if n_pos == 0:
                continue
            for pos in range(n_pos):
                total = np.float32(0.0)
                for term in range(n_terms):
                    code = 0
                    for offset in range(kmer):
                        src = pos + (window_size - 1 - (term + offset))
                        encoded = 4
                        if 0 <= src < length:
                            base = int(values[row, src])
                            encoded = 4 if base == 4 else 3 - base
                        code = code * 5 + encoded
                    total += model_rows[code, term]
                scores[row, pos] = total
                mask[row, pos] = True
        return scores, mask

    for fname, target_order in [("myog.ihbcp", 1), ("myog.ihbcp", 0)]:
        path = str(EXAMPLES_DIR / fname)
        rep = read_bamm(path, target_order)
        model_rows = rep.reshape(-1, rep.shape[-1])
        motif_len = rep.shape[-1]
        kmer = target_order + 1
        base = fname.replace(".ihbcp", "")

        fwd_scores, fwd_mask = _scan_forward(values, lengths, model_rows, kmer, motif_len)
        fwd_id = f"bamm_scan_forward_{base}_order{target_order}_seed42"
        fixtures.append(
            {
                "id": fwd_id,
                "description": f"Forward BaMM scan of {fname} (order={target_order}) on 50 random sequences (seed=42, len=200)",
                "arrays": {
                    "values": _save_npy(fwd_id + "__values", fwd_scores),
                    "mask": _save_npy(fwd_id + "__mask", fwd_mask),
                    "lengths": _save_npy(fwd_id + "__lengths", lengths),
                },
                "metadata": {
                    "n_sequences": 50,
                    "seq_length": 200,
                    "seed": 42,
                    "motif_length": motif_len,
                    "order": target_order,
                    "kmer": kmer,
                    "padding_value": 0.0,
                },
            }
        )

        rev_scores, rev_mask = _scan_reverse(values, lengths, model_rows, kmer, motif_len)
        rev_id = f"bamm_scan_reverse_{base}_order{target_order}_seed42"
        fixtures.append(
            {
                "id": rev_id,
                "description": f"Reverse BaMM scan of {fname} (order={target_order}) on 50 random sequences (seed=42, len=200)",
                "arrays": {
                    "values": _save_npy(rev_id + "__values", rev_scores),
                    "mask": _save_npy(rev_id + "__mask", rev_mask),
                    "lengths": _save_npy(rev_id + "__lengths", lengths),
                },
                "metadata": {
                    "n_sequences": 50,
                    "seq_length": 200,
                    "seed": 42,
                    "motif_length": motif_len,
                    "order": target_order,
                    "kmer": kmer,
                    "padding_value": 0.0,
                },
            }
        )

    # 4. Score bounds for BaMM
    for fname, target_order in [("myog.ihbcp", 1), ("myog.ihbcp", 0), ("gata2.ihbcp", 2)]:
        path = str(EXAMPLES_DIR / fname)
        rep = read_bamm(path, target_order)
        min_score = float(rep.min(axis=tuple(range(rep.ndim - 1))).sum())
        max_score = float(rep.max(axis=tuple(range(rep.ndim - 1))).sum())
        base = fname.replace(".ihbcp", "")
        fixtures.append(
            {
                "id": f"bamm_score_bounds_{base}_order{target_order}",
                "description": f"Theoretical score bounds for {fname} BaMM order={target_order}",
                "metadata": {"min_score": min_score, "max_score": max_score, "order": target_order},
            }
        )

    return fixtures


def generate_cli_fixtures() -> list[dict]:
    """Fixtures for CLI output comparison (scores only, not subprocess)."""
    fixtures = []

    # Motif self-comparison CLI JSON (simulated)
    model = read_model(str(EXAMPLES_DIR / "pif4.meme"), "pwm")
    config = create_comparator_config(metric="pcc")
    result = compare(model, model, "motif", config)
    cli_json = result.to_dict()

    fixtures.append(
        {
            "id": "cli_motif_self_pif4_pcc",
            "description": "CLI JSON output for motif self-comparison pif4 pcc",
            "json": _save_json("cli_motif_self_pif4_pcc", cli_json),
            "metadata": {"keys": sorted(cli_json.keys())},
        }
    )

    return fixtures


def main() -> None:
    _ensure_dir()

    import subprocess

    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent)
        .decode()
        .strip()
    )

    all_fixtures: list[dict] = []
    all_fixtures.extend(generate_pwm_parsing_fixtures())
    all_fixtures.extend(generate_sequence_fixtures())
    all_fixtures.extend(generate_reverse_complement_fixtures())
    all_fixtures.extend(generate_scan_fixtures())
    all_fixtures.extend(generate_normalization_fixtures())
    all_fixtures.extend(generate_motif_alignment_fixtures())
    all_fixtures.extend(generate_score_profile_fixtures())
    all_fixtures.extend(generate_profile_comparison_fixtures())
    all_fixtures.extend(generate_site_reconstruction_fixtures())
    all_fixtures.extend(generate_gev_fixtures())
    all_fixtures.extend(generate_bamm_fixtures())
    all_fixtures.extend(generate_cli_fixtures())

    manifest = {
        "format": "mimosa-compatibility-fixtures",
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_commit": git_commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "n_fixtures": len(all_fixtures),
        "fixtures": all_fixtures,
    }

    manifest_path = FIXTURE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Generated {len(all_fixtures)} fixtures in {FIXTURE_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Python commit: {git_commit}")
    print(f"NumPy: {np.__version__}, SciPy: {scipy.__version__}")


if __name__ == "__main__":
    main()
