"""Readers and writers for MEME, PFM, SiteGA, BaMM, and DIST motif files."""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

_SITEGA_EPS = 1e-9
_MEME_MIN_MOTIF_FIELDS = 2
_PFM_NUCLEOTIDE_ROWS = {4, 5}
_SITEGA_SEGMENT_FIELDS = 5
_DINUCLEOTIDE_LENGTH = 2
_MATRIX_NDIM = 2


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


def read_sitega(path: str) -> tuple[np.ndarray, str, int, float, float]:
    """Parse SiteGA output file and return the motif matrix with metadata."""
    converter = {"A": 0, "C": 1, "G": 2, "T": 3}
    with open(path) as file:
        name = file.readline().strip()
        if not name:
            raise ValueError(f"Malformed SiteGA file {path}: missing model name.")
        _number_of_lpd = int(file.readline().strip().split()[0])
        length = int(file.readline().strip().split()[0])
        if length <= 0:
            raise ValueError(f"Malformed SiteGA file {path}: model length must be positive.")
        minimum = float(file.readline().strip().split()[0])
        maximum = float(file.readline().strip().split()[0])
        sitega = np.zeros((5, 5, length), dtype=np.float32)
        for line_number, line in enumerate(file, start=6):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != _SITEGA_SEGMENT_FIELDS:
                raise ValueError(f"Malformed SiteGA file {path}: line {line_number} must contain 5 fields.")
            start, stop, value, _, dinucleotide = parts
            dinucleotide = dinucleotide.upper()
            if len(dinucleotide) != _DINUCLEOTIDE_LENGTH or any(
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
    return np.array(sitega, dtype=np.float32), name, length, minimum, maximum


def _parse_bamm_position_block(raw_block: str, filepath: str) -> list[np.ndarray]:
    """Parse one BaMM position block into order-specific arrays."""
    stripped_lines = [line.strip() for line in raw_block.strip().split("\n")]
    valid_lines = [line for line in stripped_lines if line and not line.startswith("#")]
    block_arrays = []
    for line in valid_lines:
        parts = line.split()
        if not parts:
            continue
        arr = np.array([float(x) for x in parts], dtype=np.float32)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"Invalid BaMM values in {filepath}")
        block_arrays.append(arr)
    return block_arrays


def parse_file_content(filepath: str) -> tuple[dict[int, list[np.ndarray]], int, int]:
    """Parse BaMM file content, ignoring comments starting with '#'."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    with open(filepath, "r") as f:
        raw_text = f.read()

    raw_blocks = raw_text.strip().split("\n\n")
    clean_blocks_data = []

    for raw_block in raw_blocks:
        block_arrays = _parse_bamm_position_block(raw_block, filepath)
        if not block_arrays:
            continue
        clean_blocks_data.append(block_arrays)

    if not clean_blocks_data:
        raise ValueError(f"No valid data found in {filepath}")

    num_positions = len(clean_blocks_data)
    if num_positions <= 0:
        raise ValueError(f"No valid positions found in {filepath}")
    max_order = len(clean_blocks_data[0]) - 1

    data_by_order: dict[int, list[np.ndarray]] = {}
    for k in range(max_order + 1):
        data_by_order[k] = []
        expected_width = 4 ** (k + 1)
        for pos_idx in range(num_positions):
            if len(clean_blocks_data[pos_idx]) <= k:
                raise ValueError(f"Inconsistent orders in block {pos_idx}")
            values = clean_blocks_data[pos_idx][k]
            if values.size != expected_width:
                raise ValueError(
                    f"Inconsistent BaMM order {k} width in block {pos_idx}: "
                    f"expected {expected_width}, got {values.size}"
                )
            data_by_order[k].append(values)

    return data_by_order, max_order, num_positions


def read_bamm(motif_path: str, target_order: int) -> np.ndarray:
    """Read a BaMM motif and convert it to log-odds against a uniform background."""

    motif_raw, max_order_file, motif_length = parse_file_content(motif_path)
    if target_order > max_order_file:
        target_order = max_order_file
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Target order {target_order} exceeds file max order {max_order_file}, target order set as max order"
        )

    acgt_slices = []

    for pos in range(motif_length):
        current_k = min(pos, target_order)

        p_motif = motif_raw[current_k][pos]
        uniform_bg = np.full_like(p_motif, 0.25 ** (current_k + 1), dtype=np.float32)

        epsilon = 1e-10
        log_odds = np.log((p_motif + epsilon) / (uniform_bg + epsilon))

        shape_k = [4] * (current_k + 1)
        tensor_k = log_odds.reshape(shape_k)

        if current_k < target_order:
            missing_dims = target_order - current_k
            expand_shape = [1] * missing_dims + shape_k
            tensor_expanded = tensor_k.reshape(expand_shape)
            target_shape_4 = [4] * (target_order + 1)
            tensor_final = np.broadcast_to(tensor_expanded, target_shape_4).copy()
        else:
            tensor_final = tensor_k

        acgt_slices.append(tensor_final)

    acgt_tensor = np.stack(acgt_slices, axis=-1)

    reduce_axes = tuple(range(target_order + 1))
    min_scores_per_pos = np.min(acgt_tensor, axis=reduce_axes)

    new_shape = [5] * (target_order + 1) + [motif_length]

    final_tensor = np.ones(new_shape, dtype=np.float32) * min_scores_per_pos

    slice_objs = [slice(0, 4)] * (target_order + 1) + [slice(None)]
    final_tensor[tuple(slice_objs)] = acgt_tensor

    return np.array(final_tensor, dtype=np.float32)


def write_sitega(model, path: str) -> None:
    """Write SiteGA motif to a .mat file."""
    from .scanning import get_score_bounds

    sitega_matrix = model.representation
    minimum, maximum = get_score_bounds(model)
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


def write_pfm(pfm: np.ndarray, name: str, _length: int, path: str) -> None:
    """Write a Position Frequency Matrix to a file."""
    with open(path, "w") as f:
        f.write(f">{name}\n")

        np.savetxt(f, pfm.T, fmt="%.6f", delimiter="\t")


def read_pfm(path: str) -> tuple[np.ndarray, int]:
    """Read a Position Frequency Matrix (PFM) from a file."""
    raw = np.loadtxt(path, comments=">", dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != _MATRIX_NDIM:
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


def write_dist(threshold_table: np.ndarray, max_score, min_score, path: str) -> None:
    """Write the threshold table of motif to a DIST formatted file."""
    score_range = float(max_score) - float(min_score)
    if score_range <= 0.0:
        raise ValueError("max_score must be greater than min_score when writing DIST tables.")
    table = np.array(threshold_table, dtype=np.float64, copy=True)
    table[:, 0] = (table[:, 0] - float(min_score)) / score_range
    with open(path, "w") as fname:
        np.savetxt(fname, table, fmt="%.18f", delimiter="\t", newline="\n", footer="", comments="", encoding=None)
