"""SiteGA motif readers and writers."""

from __future__ import annotations

import itertools

import numpy as np

_SITEGA_EPS = 1e-9


def read_sitega(path: str) -> tuple[np.ndarray, str, int]:
    """Parse SiteGA output file and return the motif matrix with metadata."""
    converter = {"A": 0, "C": 1, "G": 2, "T": 3}
    segment_fields = 5
    dinucleotide_length = 2
    with open(path) as file:
        name = file.readline().strip()
        if not name:
            raise ValueError(f"Malformed SiteGA file {path}: missing model name.")
        _number_of_lpd = int(file.readline().strip().split()[0])
        length = int(file.readline().strip().split()[0])
        if length <= 0:
            raise ValueError(f"Malformed SiteGA file {path}: model length must be positive.")
        file.readline()  # skip row
        file.readline()  # skip row
        sitega = np.zeros((5, 5, length), dtype=np.float32)
        for line_number, line in enumerate(file, start=6):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != segment_fields:
                raise ValueError(f"Malformed SiteGA file {path}: line {line_number} must contain 5 fields.")
            start, stop, value, _, dinucleotide = parts
            dinucleotide = dinucleotide.upper()
            if len(dinucleotide) != dinucleotide_length or any(
                nucleotide not in converter for nucleotide in dinucleotide
            ):
                raise ValueError(f"Malformed SiteGA file {path}: invalid dinucleotide {dinucleotide!r}.")
            start_index = int(start)
            stop_index = int(stop)
            if start_index < 0 or stop_index < start_index or stop_index >= length:
                raise ValueError(
                    f"Malformed SiteGA file {path}: range {start_index}-{stop_index} is outside model length {length}."
                )
            nuc_1, nuc_2 = converter[dinucleotide[0]], converter[dinucleotide[1]]
            number_of_positions = stop_index - start_index + 1
            for index in range(start_index, stop_index + 1):
                sitega[nuc_1][nuc_2][index] += float(value) / number_of_positions
    return np.array(sitega, dtype=np.float32), name, length


def write_sitega(model, path: str) -> None:
    """Write SiteGA motif to a .mat file."""
    from mimosa.scanning import score_bounds_from_representation

    sitega_matrix = model.representation
    minimum, maximum = score_bounds_from_representation(np.asarray(sitega_matrix))
    converter = {0: "A", 1: "C", 2: "G", 3: "T"}
    dinuc_map = {"".join(dinuc): index for index, dinuc in enumerate(itertools.product("acgt", repeat=2))}

    segments = []

    for nuc1 in range(4):
        for nuc2 in range(4):
            if np.all(np.abs(sitega_matrix[nuc1, nuc2, :]) <= _SITEGA_EPS):
                continue

            dinucleotide = converter[nuc1] + converter[nuc2]
            pos = 0

            while pos < model.length:
                while pos < model.length and abs(sitega_matrix[nuc1, nuc2, pos]) <= _SITEGA_EPS:
                    pos += 1

                if pos >= model.length:
                    break

                start_pos = pos
                current_val = sitega_matrix[nuc1, nuc2, pos]

                while pos + 1 < model.length and abs(sitega_matrix[nuc1, nuc2, pos + 1] - current_val) < _SITEGA_EPS:
                    pos += 1

                segments.append({"start": start_pos, "stop": pos, "val": current_val, "dinucl": dinucleotide})

                pos += 1

    lpd_count = len(segments)

    with open(path, "w") as f:
        f.write(f"{model.name}\n")
        f.write(f"{lpd_count}\tLPD count\n")
        f.write(f"{model.length}\tModel length\n")
        f.write(f"{minimum:.12f}\tMinimum\n")
        f.write(f"{maximum:.12f}\tRazmah\n")

        for seg in segments:
            range_length = seg["stop"] - seg["start"] + 1
            total_value = seg["val"] * range_length
            dinuc_index = dinuc_map[seg["dinucl"].lower()]
            f.write(f"{seg['start']}\t{seg['stop']}\t{total_value:.12f}\t{dinuc_index}\t{seg['dinucl'].lower()}\n")
