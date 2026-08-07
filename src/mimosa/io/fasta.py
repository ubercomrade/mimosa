"""FASTA and score-profile readers."""

from __future__ import annotations

import numpy as np

from ..arrays import EncodedSequences, RaggedArray, _ENCODE_TABLE
from ..errors import ModelFormatError
from ..profiles.prepared import ScoreProfile

MAX_FASTA_SEQUENCES = 1_000_000
MAX_FASTA_SEQUENCE_LENGTH = 100_000_000
MAX_FASTA_LINE_LENGTH = 1_000_000

MAX_SCORE_FILE_BYTES = 256 * 1024**2
MAX_SCORE_LINE_BYTES = 4 * 1024**2
MAX_SCORE_ROWS = 1_000_000
MAX_SCORE_ELEMENTS = 100_000_000


def read_fasta(path, max_sequences=MAX_FASTA_SEQUENCES):
    if max_sequences < 1:
        raise ValueError("max_sequences must be at least 1.")
    rows = []
    names = []
    current_seq = bytearray()
    current_name = ""
    has_current = False
    n_sequences = 0
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
                    if n_sequences >= max_sequences:
                        raise ModelFormatError(
                            path, f"exceeded max_sequences limit {max_sequences}."
                        )
                    rows.append(bytes(current_seq))
                    names.append(current_name)
                    n_sequences += 1
                    current_seq = bytearray()
                header = stripped[1:].strip()
                current_name = header.split()[0] if header else ""
                has_current = True
            else:
                if not has_current:
                    raise ModelFormatError(path, "sequence data before header line.")
                current_seq.extend(stripped.encode("ascii", errors="replace"))
                if len(current_seq) > MAX_FASTA_SEQUENCE_LENGTH:
                    raise ModelFormatError(
                        path, f"sequence exceeds length limit {MAX_FASTA_SEQUENCE_LENGTH}."
                    )
    if has_current:
        if n_sequences >= max_sequences:
            raise ModelFormatError(path, f"exceeded max_sequences limit {max_sequences}.")
        rows.append(bytes(current_seq))
        names.append(current_name)
        n_sequences += 1
    if not rows:
        raise ModelFormatError(path, "no sequences found in FASTA file.")
    data = np.frombuffer(b"".join(rows), dtype=np.uint8)
    encoded = _ENCODE_TABLE[data]
    offsets = np.zeros(n_sequences + 1, dtype=np.int64)
    for i, r in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(r)
    return EncodedSequences(encoded, offsets), tuple(names)


def read_sequences(path, **kwargs):
    return read_fasta(path, **kwargs)


def read_scores(path):
    file = str(path)
    rows = []
    current_values = []
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
                    rows.append(current_values)
                if len(rows) >= MAX_SCORE_ROWS:
                    raise ModelFormatError(file, "score row count exceeds the limit.")
                current_values = []
                seen_header = True
                continue
            if not seen_header:
                raise ModelFormatError(file, "score values require a header.")
            cleaned = stripped.replace(",", " ")
            for token in cleaned.split():
                if not token:
                    continue
                try:
                    value = float(token)
                except ValueError:
                    raise ModelFormatError(file, f"invalid score value: '{token}'.")
                if not np.isfinite(value):
                    raise ModelFormatError(file, "score values must be finite.")
                elements += 1
                if elements > MAX_SCORE_ELEMENTS:
                    raise ModelFormatError(file, "score element count exceeds the limit.")
                current_values.append(value)
    if seen_header:
        rows.append(current_values)
    if len(rows) > MAX_SCORE_ROWS:
        raise ModelFormatError(file, "score row count exceeds the limit.")
    if not rows:
        raise ModelFormatError(file, "score file contains no profiles.")
    name = file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return ScoreProfile(name, RaggedArray.from_rows(rows))
