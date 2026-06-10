"""BaMM motif readers."""

from __future__ import annotations

import logging
import os

import numpy as np


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
