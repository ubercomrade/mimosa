#!/usr/bin/env python3
"""Generate frozen Python oracle fixtures for Slim (Jstacs GenDisMix) models.

Mirrors the Dimont fixture set: parse representations, score bounds, and
forward/reverse scanning on the shared seed=42 random 5-ary batch.

Run from the repository root:
    .venv/bin/python scripts/generate_slim_fixtures.py

Outputs are written to tests/fixtures/compatibility/ and the corresponding
manifest entries are printed to stdout for manual insertion into manifest.json.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures" / "compatibility"
SLIM_DIR = REPO / "tests" / "fixtures" / "models" / "slim"

SLIM_FILES = [
    "example-model-1.xml",
    "PEAKS036274_FOXA1_P35582_MACS2-model-2.xml",
    "PEAKS038885_CEBPB_P28033_MACS2-model-1.xml",
    "PEAKS038885_CEBPB_P28033_MACS2-model-2.xml",
]
SCAN_FILES = ["example-model-1.xml", "PEAKS036274_FOXA1_P35582_MACS2-model-2.xml"]


def save_npy(name: str, arr: np.ndarray) -> dict:
    path = FIXTURE_DIR / f"{name}.npy"
    np.save(path, arr)
    return {"path": str(path.relative_to(REPO)), "dtype": str(arr.dtype), "shape": list(arr.shape)}


def scan_forward(values, lengths, model_rows, kmer, motif_len):
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


def scan_reverse(values, lengths, model_rows, kmer, motif_len):
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


def main():
    sys.path.insert(0, str(REPO / "src"))
    from mimosa.io.xml import read_slim

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the same random batch as Dimont (seed=42, n=50, len=200, 5-ary).
    values = np.load(FIXTURE_DIR / "dimont_scan_input_seed42__values.npy")
    lengths = np.load(FIXTURE_DIR / "dimont_scan_input_seed42__lengths.npy")

    fixtures = []

    # 1. Parse Slim and save flattened representation for each file.
    for fname in SLIM_FILES:
        path = str(SLIM_DIR / fname)
        rep, length, span = read_slim(path)
        flat = rep.reshape(-1, length)  # C-order, matches scanning
        base = fname.replace(".xml", "")
        fid = f"slim_parse_{base}"
        fixtures.append(
            {
                "id": fid,
                "description": f"Parse {fname} to flattened (5^(span+1), length) representation",
                "arrays": {"representation": save_npy(fid + "__representation", flat)},
                "metadata": {
                    "name": base,
                    "motif_length": int(length),
                    "span": int(span),
                    "kmer": int(span + 1),
                    "shape": list(flat.shape),
                },
            }
        )

    # 2. Score bounds for Slim.
    for fname in SLIM_FILES:
        path = str(SLIM_DIR / fname)
        rep, length, span = read_slim(path)
        rep_arr = np.asarray(rep, dtype=np.float32)
        min_score = float(rep_arr.min(axis=tuple(range(rep_arr.ndim - 1))).sum())
        max_score = float(rep_arr.max(axis=tuple(range(rep_arr.ndim - 1))).sum())
        base = fname.replace(".xml", "")
        fid = f"slim_score_bounds_{base}"
        fixtures.append(
            {
                "id": fid,
                "description": f"Theoretical score bounds for {fname} Slim",
                "metadata": {"min_score": min_score, "max_score": max_score, "span": int(span)},
            }
        )

    # 3. Shared scan input (reuse dimont input, but alias for slim for clarity).
    seq_fid = "slim_scan_input_seed42"
    fixtures.append(
        {
            "id": seq_fid,
            "description": "Input sequences for Slim scan fixtures (seed=42, n=50, len=200, 5-ary encoding)",
            "arrays": {
                "values": save_npy(seq_fid + "__values", values),
                "lengths": save_npy(seq_fid + "__lengths", lengths),
            },
            "metadata": {"n_sequences": 50, "seq_length": 200, "seed": 42, "padding_value": 4},
        }
    )

    # 4. Forward and reverse scanning for two files.
    for fname in SCAN_FILES:
        path = str(SLIM_DIR / fname)
        rep, length, span = read_slim(path)
        model_rows = rep.reshape(-1, length)
        motif_len = length
        kmer = span + 1
        base = fname.replace(".xml", "")

        fwd_scores, fwd_mask = scan_forward(values, lengths, model_rows, kmer, motif_len)
        fwd_id = f"slim_scan_forward_{base}_seed42"
        fixtures.append(
            {
                "id": fwd_id,
                "description": f"Forward Slim scan of {fname} on 50 random sequences (seed=42, len=200)",
                "arrays": {
                    "values": save_npy(fwd_id + "__values", fwd_scores),
                    "mask": save_npy(fwd_id + "__mask", fwd_mask),
                    "lengths": save_npy(fwd_id + "__lengths", lengths),
                },
                "metadata": {
                    "n_sequences": 50,
                    "seq_length": 200,
                    "seed": 42,
                    "motif_length": int(motif_len),
                    "span": int(span),
                    "kmer": int(kmer),
                    "padding_value": 0.0,
                },
            }
        )

        rev_scores, rev_mask = scan_reverse(values, lengths, model_rows, kmer, motif_len)
        rev_id = f"slim_scan_reverse_{base}_seed42"
        fixtures.append(
            {
                "id": rev_id,
                "description": f"Reverse Slim scan of {fname} on 50 random sequences (seed=42, len=200)",
                "arrays": {
                    "values": save_npy(rev_id + "__values", rev_scores),
                    "mask": save_npy(rev_id + "__mask", rev_mask),
                    "lengths": save_npy(rev_id + "__lengths", lengths),
                },
                "metadata": {
                    "n_sequences": 50,
                    "seq_length": 200,
                    "seed": 42,
                    "motif_length": int(motif_len),
                    "span": int(span),
                    "kmer": int(kmer),
                    "padding_value": 0.0,
                },
            }
        )

    # Print manifest entries for manual insertion.
    print("=== MANIFEST ENTRIES (JSON) ===")
    print(json.dumps(fixtures, indent=2, sort_keys=True))
    print(f"\nGenerated {len(fixtures)} Slim fixtures in {FIXTURE_DIR}")


if __name__ == "__main__":
    main()