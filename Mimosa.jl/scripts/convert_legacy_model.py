#!/usr/bin/env python3
"""Convert legacy Mimosa model files (pickle/joblib) to the portable Mimosa.jl bundle format.

This script reads legacy serialized model files from the Python MIMOSA project
and writes them to the language-neutral directory bundle format used by Mimosa.jl.

SECURITY WARNING:
  pickle and joblib files can execute arbitrary Python code on load.
  Only run this script on files from trusted sources.
  Never use it on untrusted or user-uploaded files.

Usage:
  python convert_legacy_model.py <input.pkl> <output_dir> [--type pwm|bamm|sitega|dimont|slim]
  python convert_legacy_model.py <input.joblib> <output_dir> [--type pwm]

The output directory will contain:
  manifest.toml  - metadata (format, kind, name, shape, checksums)
  data/          - NPY binary blobs

Requirements:
  pip install numpy joblib

Exit codes:
  0 = success
  1 = usage error
  2 = runtime error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_npy_2d(path: Path, data) -> None:
    """Write a 2D float32 array in NumPy .npy format (row-major, little-endian)."""
    import numpy as np

    arr = np.ascontiguousarray(data, dtype=np.float32)
    np.save(str(path), arr)


def detect_model_type(obj) -> str:
    """Attempt to detect model type from a deserialized Python object."""
    if hasattr(obj, "type_key"):
        return obj.type_key
    if hasattr(obj, "representation"):
        rep = obj.representation
        if hasattr(rep, "shape"):
            shape = rep.shape
            if len(shape) == 2:
                if shape[0] == 5:
                    return "pwm"
                if shape[0] in (4,):
                    return "pwm"
                if shape[0] == 25:
                    return "sitega"
                # Check power of 4 for BaMM/Dimont/Slim
                n = shape[0]
                while n > 1 and n % 4 == 0:
                    n //= 4
                if n == 1:
                    return "bamm"
    raise ValueError("Cannot detect model type. Please specify --type explicitly.")


def extract_model_data(obj, model_type: str):
    """Extract name and representation matrix from a legacy model object."""
    name = getattr(obj, "name", "unknown")

    if model_type == "pwm":
        # PWM: weights matrix (5, W) and background
        rep = obj.representation
        if hasattr(rep, "shape") and rep.shape[0] == 4:
            # Need to extend with N row
            import numpy as np
            n_row = np.min(rep, axis=0, keepdims=True)
            rep = np.vstack([rep, n_row])
        background = getattr(obj, "background", None)
        return name, rep, background
    elif model_type in ("bamm", "sitega", "dimont", "slim"):
        rep = obj.representation
        return name, rep, None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def write_bundle(output_dir: Path, model_type: str, name: str, representation, background=None) -> None:
    """Write a Mimosa.jl portable bundle."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine array name
    array_name = "weights" if model_type == "pwm" else "frequencies" if model_type == "pfm" else "representation"

    npy_path = data_dir / f"{array_name}.npy"
    write_npy_2d(npy_path, representation)
    checksum = sha256_file(npy_path)

    manifest = {
        "format": "mimosa",
        "format_version": 1,
        "kind": model_type,
        "name": name,
        "dtype": "<f4",
        "shape": [representation.shape[0], representation.shape[1]],
        "layout": "row_major",
        "convention": "axes: (base, position) for matrix models, (context_code, position) for higher-order",
        "provenance": {"tool": "convert_legacy_model.py", "version": "0.1.0"},
        "arrays": {
            array_name: {
                "file": f"data/{array_name}.npy",
                "dtype": "<f4",
                "shape": [representation.shape[0], representation.shape[1]],
                "checksum": f"sha256:{checksum}",
            }
        },
    }

    if background is not None:
        manifest["background"] = list(background)

    import tomli_w

    manifest_path = output_dir / "manifest.toml"
    with open(manifest_path, "wb") as f:
        tomli_w.dump(manifest, f)

    print(json.dumps({"input": str(output_dir), "type": model_type, "name": name}))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert legacy Mimosa model files to portable Mimosa.jl format.",
        epilog="SECURITY: Only use on trusted files. pickle/joblib can execute arbitrary code.",
    )
    parser.add_argument("input", help="Input legacy model file (.pkl or .joblib)")
    parser.add_argument("output", help="Output directory for the Mimosa.jl bundle")
    parser.add_argument("--type", choices=["pwm", "bamm", "sitega", "dimont", "slim"],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--trusted-input", action="store_true",
                        help="Acknowledge that this file is from a trusted source")

    args = parser.parse_args()

    if not args.trusted_input:
        print("ERROR: This script loads pickle/joblib files which can execute arbitrary code.", file=sys.stderr)
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
        print("ERROR: joblib is required. Install with: pip install joblib numpy", file=sys.stderr)
        return 2

    try:
        obj = joblib.load(input_path)
    except Exception as e:
        print(f"ERROR: failed to load {input_path}: {e}", file=sys.stderr)
        return 2

    model_type = args.type or detect_model_type(obj)
    name, representation, background = extract_model_data(obj, model_type)

    write_bundle(output_path, model_type, name, representation, background)
    return 0


if __name__ == "__main__":
    sys.exit(main())