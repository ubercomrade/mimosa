"""Flat-buffer ragged storage for encoded sequences and score tracks."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .errors import InvariantError

N_CODE = 4

_ENCODE_TABLE = np.full(256, N_CODE, dtype=np.uint8)
_ENCODE_TABLE[ord("A")] = 0
_ENCODE_TABLE[ord("C")] = 1
_ENCODE_TABLE[ord("G")] = 2
_ENCODE_TABLE[ord("T")] = 3
_ENCODE_TABLE[ord("a")] = 0
_ENCODE_TABLE[ord("c")] = 1
_ENCODE_TABLE[ord("g")] = 2
_ENCODE_TABLE[ord("t")] = 3


def _validate_ragged_offsets(offsets, data_len):
    if offsets.size == 0:
        raise ValueError("offsets must not be empty")
    if offsets[0] != 0:
        raise ValueError(f"offsets[0] must be 0, got {offsets[0]}.")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be non-decreasing.")
    if offsets[-1] != data_len:
        raise ValueError(
            f"offsets[-1] must be len(data)={data_len}, got {offsets[-1]}."
        )


def _validate_raw_encoded_data(data):
    if data.size == 0:
        return
    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("encoded data must have an integer dtype.")
    invalid = (data < 0) | (data > N_CODE)
    if np.any(invalid):
        bad = int(np.flatnonzero(invalid)[0])
        raise InvariantError(
            f"invalid encoded base {data[bad]!r} at index {bad}; "
            "valid codes are 0..4 (A,C,G,T,N)."
        )


def _from_rows(cls, rows, dtype):
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    for i, row in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(row)
    data = np.empty(int(offsets[-1]), dtype=dtype)
    for i, row in enumerate(rows):
        data[offsets[i] : offsets[i + 1]] = row
    return cls(data, offsets)


class EncodedSequences:
    """All sequences in one uint8 buffer plus int64 offsets (zero-based)."""

    __slots__ = ("data", "offsets")

    def __init__(self, data, offsets):
        raw_data = np.asarray(data)
        raw_offsets = np.asarray(offsets)
        if raw_data.ndim != 1:
            raise ValueError("encoded data must be one-dimensional.")
        if raw_offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional.")
        if not np.issubdtype(raw_offsets.dtype, np.integer):
            raise TypeError("offsets must have an integer dtype.")
        _validate_raw_encoded_data(raw_data)
        data = np.array(raw_data, dtype=np.uint8, order="C", copy=True)
        offsets = np.array(raw_offsets, dtype=np.int64, order="C", copy=True)
        _validate_ragged_offsets(offsets, data.size)
        data.setflags(write=False)
        offsets.setflags(write=False)
        self.data = data
        self.offsets = offsets

    @classmethod
    def from_rows(cls, rows):
        rows = list(rows)
        for row in rows:
            _validate_raw_encoded_data(np.asarray(row))
        return _from_rows(cls, rows, np.uint8)

    @classmethod
    def from_strings(cls, strings):
        rows = [encode_sequence(s) for s in strings]
        return cls.from_rows(rows)

    def __len__(self):
        return self.offsets.size - 1

    def __getitem__(self, i):
        return self.data[self.offsets[i] : self.offsets[i + 1]]

    def __eq__(self, other):
        return (
            isinstance(other, EncodedSequences)
            and np.array_equal(self.offsets, other.offsets)
            and np.array_equal(self.data, other.data)
        )

    def __repr__(self):
        return f"EncodedSequences({len(self)} sequences, {self.data.size} bytes)"


def encode_sequence(s):
    return _ENCODE_TABLE[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]


def reverse_complement(seq):
    """Reverse complement of an encoded sequence (N stays N)."""
    rc = seq[::-1].copy()
    mask = rc != N_CODE
    rc[mask] = 3 - rc[mask]
    return rc


class RaggedArray:
    """Flat offset-based ragged storage for float32 score tracks."""

    __slots__ = ("data", "offsets")

    def __init__(self, data, offsets):
        raw_data = np.asarray(data)
        raw_offsets = np.asarray(offsets)
        if raw_data.ndim != 1:
            raise ValueError("ragged data must be one-dimensional.")
        if raw_offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional.")
        if not np.issubdtype(raw_offsets.dtype, np.integer):
            raise TypeError("offsets must have an integer dtype.")
        data = np.ascontiguousarray(raw_data, dtype=np.float32)
        offsets = np.ascontiguousarray(raw_offsets, dtype=np.int64)
        _validate_ragged_offsets(offsets, data.size)
        self.data = data
        self.offsets = offsets

    @classmethod
    def from_rows(cls, rows):
        return _from_rows(cls, rows, np.float32)

    def __len__(self):
        return self.offsets.size - 1

    def __getitem__(self, i):
        return self.data[self.offsets[i] : self.offsets[i + 1]]

    def __eq__(self, other):
        return (
            isinstance(other, RaggedArray)
            and np.array_equal(self.offsets, other.offsets)
            and np.array_equal(self.data, other.data)
        )

    def __repr__(self):
        return f"RaggedArray({len(self)} rows, {self.data.size} elements)"


@dataclass(slots=True)
class StrandPair:
    """Forward and reverse RaggedArray values; may share the same object."""

    forward: RaggedArray
    reverse: RaggedArray
