"""FASTA and score-profile readers."""

from __future__ import annotations

import re

import numpy as np

from ..arrays import EncodedSequences, RaggedArray, _ENCODE_TABLE
from ..errors import ModelFormatError
from ..profiles.prepared import ScoreProfile

MAX_FASTA_SEQUENCES = 1_000_000
MAX_FASTA_SEQUENCE_LENGTH = 100_000_000
MAX_FASTA_LINE_LENGTH = 1_000_000
MAX_FASTA_TOTAL_BASES = 1_000_000_000

MAX_SCORE_LINE_BYTES = 4 * 1024**2
MAX_SCORE_ROWS = 1_000_000
MAX_SCORE_ELEMENTS = 100_000_000

_SCORE_TOKEN = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\Z"
)


def read_fasta(path, max_sequences=MAX_FASTA_SEQUENCES):
    if max_sequences < 1:
        raise ValueError("max_sequences must be at least 1.")
    names = []
    data = bytearray()
    offsets = [0]
    current_name = ""
    has_current = False
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if len(line) > MAX_FASTA_LINE_LENGTH:
                raise ModelFormatError(
                    path, f"line exceeds length limit {MAX_FASTA_LINE_LENGTH}."
                )
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0] == ">":
                if has_current:
                    names.append(current_name)
                    offsets.append(len(data))
                if len(names) >= max_sequences:
                    raise ModelFormatError(
                        path, f"exceeded max_sequences limit {max_sequences}."
                    )
                header = stripped[1:].strip()
                current_name = header.split()[0] if header else ""
                has_current = True
            else:
                if not has_current:
                    raise ModelFormatError(path, "sequence data before header line.")
                sequence_bytes = stripped.encode("ascii", errors="replace")
                if len(data) - offsets[-1] + len(sequence_bytes) > MAX_FASTA_SEQUENCE_LENGTH:
                    raise ModelFormatError(
                        path, f"sequence exceeds length limit {MAX_FASTA_SEQUENCE_LENGTH}."
                    )
                if len(data) + len(sequence_bytes) > MAX_FASTA_TOTAL_BASES:
                    raise ModelFormatError(
                        path, f"total bases exceed limit {MAX_FASTA_TOTAL_BASES}."
                    )
                data.extend(sequence_bytes)
    if has_current:
        names.append(current_name)
        offsets.append(len(data))
    if not names:
        raise ModelFormatError(path, "no sequences found in FASTA file.")
    encoded = _ENCODE_TABLE[np.frombuffer(data, dtype=np.uint8)]
    return EncodedSequences(encoded, np.asarray(offsets, dtype=np.int64)), tuple(names)


def read_scores(path):
    file = str(path)
    rows = []
    current_chunks = []
    seen_header = False
    elements = 0
    with open(file, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if len(line) > MAX_SCORE_LINE_BYTES:
                raise ModelFormatError(file, "score line exceeds the size limit.")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if seen_header:
                    rows.append(_join_score_chunks(current_chunks))
                if len(rows) >= MAX_SCORE_ROWS:
                    raise ModelFormatError(file, "score row count exceeds the limit.")
                current_chunks = []
                seen_header = True
                continue
            if not seen_header:
                raise ModelFormatError(file, "score values require a header.")
            cleaned = stripped.replace(",", " ")
            tokens = cleaned.split()
            invalid = next((token for token in tokens if not _SCORE_TOKEN.fullmatch(token)), None)
            if invalid is not None:
                raise ModelFormatError(file, f"invalid score value: '{invalid}'.")
            values = np.fromstring(cleaned, dtype=np.float32, sep=" ")
            if values.size != len(tokens):
                raise ModelFormatError(file, "invalid score value.")
            if not np.all(np.isfinite(values)):
                raise ModelFormatError(file, "score values must be finite.")
            elements += values.size
            if elements > MAX_SCORE_ELEMENTS:
                raise ModelFormatError(file, "score element count exceeds the limit.")
            current_chunks.append(values)
    if seen_header:
        rows.append(_join_score_chunks(current_chunks))
    if len(rows) > MAX_SCORE_ROWS:
        raise ModelFormatError(file, "score row count exceeds the limit.")
    if not rows:
        raise ModelFormatError(file, "score file contains no profiles.")
    name = file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return ScoreProfile(name, RaggedArray.from_rows(rows))


def _join_score_chunks(chunks):
    if not chunks:
        return np.empty(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)
