"""MEME motif readers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

_MEME_MIN_MOTIF_FIELDS = 2


def _meme_length_from_header(header_line: str) -> int:
    header = header_line.strip().split()
    try:
        length_idx = header.index("w=") + 1
        return int(header[length_idx])
    except (ValueError, IndexError):
        return 0


def _read_meme_matrix_rows(handle, length: int) -> list[list[float]]:
    matrix = []
    for _ in range(length):
        row_line = handle.readline()
        row = row_line.strip().split()
        if row:
            matrix.append(list(map(float, row)))
    return matrix


def _validate_meme_matrix(matrix: np.ndarray, path: str | Path, motif_name: str, length: int) -> np.ndarray:
    """Validate and orient one MEME letter-probability matrix."""
    if length <= 0:
        raise ValueError(f"Malformed MEME file {path}: motif {motif_name!r} has invalid length.")
    if matrix.shape != (length, 4):
        raise ValueError(
            f"Malformed MEME file {path}: motif {motif_name!r} expected {length} rows with 4 columns, "
            f"got shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Malformed MEME file {path}: motif {motif_name!r} contains non-finite values.")
    return matrix.T.astype(np.float32, copy=False)


def read_meme(path: str, index: int = 0) -> Tuple[np.ndarray, Tuple[str, int], int]:
    """Read a specific motif from a MEME formatted file and return total count."""
    target_motif: np.ndarray | None = None
    target_info: Tuple[str, int] | None = None
    motif_count = 0

    with open(path) as handle:
        line = handle.readline()
        while line:
            if line.startswith("MOTIF"):
                is_target = motif_count == index
                motif_count += 1

                parts = line.strip().split()
                if len(parts) < _MEME_MIN_MOTIF_FIELDS:
                    raise ValueError(f"Malformed MEME file {path}: MOTIF line has no name.")
                name = parts[1]
                length = _meme_length_from_header(handle.readline())

                if is_target:
                    target_motif = _validate_meme_matrix(
                        np.array(_read_meme_matrix_rows(handle, length), dtype=np.float32),
                        path,
                        name,
                        length,
                    )
                    target_info = (name, length)
                else:
                    if length <= 0:
                        raise ValueError(f"Malformed MEME file {path}: motif {name!r} has invalid length.")
                    for _ in range(length):
                        handle.readline()

            line = handle.readline()

    if target_motif is None:
        if motif_count == 0:
            raise ValueError(f"No motifs found in {path}")
        raise IndexError(f"Motif index {index} out of range. File contains {motif_count} motifs.")

    if target_info is None:
        raise ValueError(f"Malformed MEME file {path}: motif metadata is missing.")

    return target_motif, target_info, motif_count


def read_meme_many(path: str | Path) -> list[tuple[np.ndarray, tuple[str, int]]]:
    """Read all motifs from one MEME file in file order."""
    motifs: list[tuple[np.ndarray, tuple[str, int]]] = []

    with open(path) as handle:
        line = handle.readline()
        while line:
            if line.startswith("MOTIF"):
                parts = line.strip().split()
                if len(parts) < _MEME_MIN_MOTIF_FIELDS:
                    raise ValueError(f"Malformed MEME file {path}: MOTIF line has no name.")
                name = parts[1]
                length = _meme_length_from_header(handle.readline())
                matrix = _validate_meme_matrix(
                    np.array(_read_meme_matrix_rows(handle, length), dtype=np.float32),
                    path,
                    name,
                    length,
                )
                motifs.append((matrix, (name, length)))
            line = handle.readline()

    if not motifs:
        raise ValueError(f"No motifs found in {path}")
    return motifs
