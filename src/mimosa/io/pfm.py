"""PFM motif readers and writers."""

from __future__ import annotations

import numpy as np

_PFM_NUCLEOTIDE_ROWS = {4, 5}


def write_pfm(pfm: np.ndarray, name: str, _length: int, path: str) -> None:
    """Write a Position Frequency Matrix to a file."""
    with open(path, "w") as f:
        f.write(f">{name}\n")

        np.savetxt(f, pfm.T, fmt="%.6f", delimiter="\t")


def read_pfm(path: str) -> tuple[np.ndarray, int]:
    """Read a Position Frequency Matrix (PFM) from a file."""
    matrix_ndim = 2
    raw = np.loadtxt(path, comments=">", dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != matrix_ndim:
        raise ValueError(f"Malformed PFM file {path}: matrix must be 2D.")
    if raw.shape[1] in _PFM_NUCLEOTIDE_ROWS:
        pfm = raw.T
    elif raw.shape[0] in _PFM_NUCLEOTIDE_ROWS:
        pfm = raw
    else:
        raise ValueError(f"Malformed PFM file {path}: one axis must contain 4 or 5 nucleotide rows.")
    if pfm.shape[1] <= 0:
        raise ValueError(f"Malformed PFM file {path}: motif length must be positive.")
    if not np.all(np.isfinite(pfm)):
        raise ValueError(f"Malformed PFM file {path}: matrix contains non-finite values.")
    pfm = pfm.astype(np.float32, copy=False)
    length = pfm.shape[1]
    return pfm, length
