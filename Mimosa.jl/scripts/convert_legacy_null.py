#!/usr/bin/env python3
"""Convert legacy Mimosa null distribution files (joblib) to the portable Mimosa.jl format.

This script reads legacy serialized null distribution files from the Python MIMOSA
project and writes them to the language-neutral TOML + NPY format used by Mimosa.jl.

SECURITY WARNING:
  joblib files can execute arbitrary Python code on load.
  Only run this script on files from trusted sources.
  Never use it on untrusted or user-uploaded files.

Usage:
  python convert_legacy_null.py <input.joblib> <output_dir> [--trusted-input]

The output directory will contain:
  manifest.toml  - null distribution metadata (strategy, metric, GEV params, checksums)
  raw_scores.npy - raw null comparison scores
  pairs.json     - contributing comparison pairs

Requirements:
  pip install numpy joblib tomli_w

Exit codes:
  0 = success
  1 = usage error
  2 = runtime error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_npy_1d(path: Path, data) -> None:
    """Write a 1D float32 array in NumPy .npy format (little-endian)."""
    import numpy as np

    arr = np.ascontiguousarray(data, dtype=np.float32)
    np.save(str(path), arr)


def extract_null_data(obj):
    """Extract null distribution data from a legacy Python object."""
    result = {}

    # Raw scores
    raw_scores = getattr(obj, "raw_scores", None)
    if raw_scores is None and hasattr(obj, "scores"):
        raw_scores = obj.scores
    if raw_scores is None:
        raise ValueError("Cannot find raw scores in null distribution object.")
    result["raw_scores"] = raw_scores

    # GEV parameters
    if hasattr(obj, "gev_params"):
        gev = obj.gev_params
        result["gev"] = {
            "shape": float(getattr(gev, "shape", getattr(gev, "c", 0.0))),
            "location": float(getattr(gev, "location", getattr(gev, "loc", 0.0))),
            "scale": float(getattr(gev, "scale", 1.0)),
            "converged": bool(getattr(gev, "converged", True)),
        }

    # Strategy and metric
    result["strategy"] = getattr(obj, "strategy", "motif")
    result["metric"] = getattr(obj, "metric", "pcc")

    # N null
    result["n_null"] = len(raw_scores)

    return result


def write_null_bundle(output_dir: Path, data: dict) -> None:
    """Write a Mimosa.jl portable null distribution bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write raw scores
    scores_path = output_dir / "raw_scores.npy"
    write_npy_1d(scores_path, data["raw_scores"])
    scores_checksum = sha256_file(scores_path)

    # Build manifest
    manifest = {
        "format": "mimosa-null",
        "format_version": 1,
        "strategy": data["strategy"],
        "metric": data["metric"],
        "n_null": data["n_null"],
        "arrays": {
            "raw_scores": {
                "file": "raw_scores.npy",
                "dtype": "<f4",
                "shape": [data["n_null"]],
                "checksum": f"sha256:{scores_checksum}",
            }
        },
    }

    if "gev" in data:
        manifest["gev"] = data["gev"]

    import tomli_w

    manifest_path = output_dir / "manifest.toml"
    with open(manifest_path, "wb") as f:
        tomli_w.dump(manifest, f)

    print(json.dumps({
        "output": str(output_dir),
        "n_null": data["n_null"],
        "strategy": data["strategy"],
        "metric": data["metric"],
    }))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert legacy Mimosa null distribution files to portable format.",
        epilog="SECURITY: Only use on trusted files. joblib can execute arbitrary code.",
    )
    parser.add_argument("input", help="Input legacy null distribution file (.joblib)")
    parser.add_argument("output", help="Output directory for the portable null bundle")
    parser.add_argument("--trusted-input", action="store_true",
                        help="Acknowledge that this file is from a trusted source")

    args = parser.parse_args()

    if not args.trusted_input:
        print("ERROR: This script loads joblib files which can execute arbitrary code.", file=sys.stderr)
        print("       Add --trusted-input to acknowledge you trust this file.", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        import joblib
    except ImportError:
        print("ERROR: joblib is required. Install with: pip install joblib numpy tomli_w", file=sys.stderr)
        return 2

    try:
        obj = joblib.load(input_path)
    except Exception as e:
        print(f"ERROR: failed to load {input_path}: {e}", file=sys.stderr)
        return 2

    data = extract_null_data(obj)
    write_null_bundle(output_path, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())