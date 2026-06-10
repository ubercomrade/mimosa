"""DIST motif writer."""

from __future__ import annotations

import numpy as np


def write_dist(threshold_table: np.ndarray, max_score, min_score, path: str) -> None:
    """Write the threshold table of motif to a DIST formatted file."""
    score_range = float(max_score) - float(min_score)
    if score_range <= 0.0:
        raise ValueError("max_score must be greater than min_score when writing DIST tables.")
    table = np.array(threshold_table, dtype=np.float64, copy=True)
    table[:, 0] = (table[:, 0] - float(min_score)) / score_range
    with open(path, "w") as fname:
        np.savetxt(fname, table, fmt="%.18f", delimiter="\t", newline="\n", footer="", comments="", encoding=None)
