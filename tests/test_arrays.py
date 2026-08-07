import numpy as np
import pytest

from mimosa import (
    EncodedSequences,
    RaggedArray,
    StrandPair,
    encode_base,
    encode_sequence,
    reverse_complement,
)
from mimosa.errors import InvariantError


class TestEncoding:
    def test_encode_base(self):
        assert encode_base(ord("A")) == 0
        assert encode_base(ord("C")) == 1
        assert encode_base(ord("G")) == 2
        assert encode_base(ord("T")) == 3
        assert encode_base(ord("a")) == 0
        assert encode_base(ord("N")) == 4
        assert encode_base(ord("R")) == 4

    def test_encode_sequence(self):
        seq = encode_sequence("ACGTNacgt")
        assert seq.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3]

    def test_reverse_complement(self):
        seq = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        rc = reverse_complement(seq)
        assert rc.tolist() == [4, 0, 1, 2, 3]
        assert reverse_complement(rc).tolist() == seq.tolist()


class TestEncodedSequences:
    def test_construction_invariants(self):
        data = np.array([0, 1, 2, 3], dtype=np.uint8)
        offsets = np.array([0, 2, 4], dtype=np.int64)
        batch = EncodedSequences(data, offsets)
        assert len(batch) == 2
        assert batch[0].tolist() == [0, 1]
        assert batch[1].tolist() == [2, 3]

    def test_offsets_must_start_at_zero(self):
        with pytest.raises(ValueError):
            EncodedSequences(np.array([0], dtype=np.uint8), np.array([1, 2], dtype=np.int64))

    def test_offsets_must_end_at_len(self):
        with pytest.raises(ValueError):
            EncodedSequences(np.array([0, 1], dtype=np.uint8), np.array([0, 1], dtype=np.int64))

    def test_invalid_codes_rejected(self):
        with pytest.raises(InvariantError):
            EncodedSequences(np.array([5], dtype=np.uint8), np.array([0, 1], dtype=np.int64))

    def test_empty_rows_valid(self):
        batch = EncodedSequences(
            np.array([], dtype=np.uint8), np.array([0, 0, 0], dtype=np.int64)
        )
        assert len(batch) == 2
        assert batch[0].size == 0

    def test_slices_are_views(self):
        data = np.array([0, 1, 2, 3], dtype=np.uint8)
        batch = EncodedSequences(data, np.array([0, 4], dtype=np.int64))
        batch[0][0] = 9
        assert data[0] == 9

    def test_from_strings(self):
        batch = EncodedSequences.from_strings(["ACGT", "", "NN"])
        assert len(batch) == 3
        assert batch[0].tolist() == [0, 1, 2, 3]
        assert batch[1].size == 0
        assert batch[2].tolist() == [4, 4]


class TestRaggedArray:
    def test_from_rows(self):
        ra = RaggedArray.from_rows([[1.0, 2.0], [], [3.0]])
        assert len(ra) == 3
        assert ra[0].tolist() == [1.0, 2.0]
        assert ra[1].size == 0
        assert ra[2].tolist() == [3.0]

    def test_float32(self):
        ra = RaggedArray.from_rows([[1.0]])
        assert ra.data.dtype == np.float32

    def test_offsets_validation(self):
        with pytest.raises(ValueError):
            RaggedArray(np.array([1.0], dtype=np.float32), np.array([1, 2], dtype=np.int64))


class TestStrandPair:
    def test_identity_preserved(self):
        ra = RaggedArray.from_rows([[1.0]])
        pair = StrandPair(ra, ra)
        assert pair.forward is pair.reverse
