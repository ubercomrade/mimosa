"""Readers for FASTA sequences and FASTA-like score tracks."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from mimosa.batches import make_score_batch, make_sequence_batch


def read_fasta(path: str | Path):
    """Read a FASTA file and return integer-encoded sequences."""

    trans_table = bytearray([4] * 256)
    for char, code in zip(b"ACGTacgt", [0, 1, 2, 3] * 2, strict=False):
        trans_table[char] = code

    sequences: List[np.ndarray] = []

    with open(path, "r") as handle:
        current_seq_bytes = bytearray()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_bytes:
                    encoded = np.frombuffer(current_seq_bytes.translate(trans_table), dtype=np.int8).copy()
                    sequences.append(encoded)
                    current_seq_bytes.clear()
            else:
                current_seq_bytes.extend(line.encode("ascii", errors="ignore"))

        if current_seq_bytes:
            encoded = np.frombuffer(current_seq_bytes.translate(trans_table), dtype=np.int8).copy()
            sequences.append(encoded)

    return make_sequence_batch(sequences)


def read_scores(path: str | Path):
    """Read FASTA-like numerical score profiles into a dense masked batch."""

    profiles: List[np.ndarray] = []
    current_values: List[float] = []

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_values:
                    profiles.append(np.asarray(current_values, dtype=np.float32))
                    current_values = []
                continue

            try:
                current_values.extend(float(token) for token in line.replace(",", " ").split())
            except ValueError as exc:
                raise ValueError(f"Invalid score value in {path}: {line}") from exc

    if current_values:
        profiles.append(np.asarray(current_values, dtype=np.float32))

    return make_score_batch(profiles)
