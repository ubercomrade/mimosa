import numpy as np
import pytest

from mimosa import MotifModel, PWM, EncodedSequences, pwm_from_pfm, reverse_complement, scan
from mimosa._kernels import ho_kmer_codes
from mimosa.errors import ModelFormatError, ModelInterfaceError
from mimosa.io.models import read_meme


def _seq(*bases):
    return np.array(bases, dtype=np.uint8)


@pytest.fixture
def pwm():
    name, pfm = read_meme("examples/foxa2.meme")
    return pwm_from_pfm(pfm, name=name)


@pytest.fixture
def batch():
    rows = [
        _seq(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3),
        _seq(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0),
        _seq(0, 0, 0, 0),
    ]
    return EncodedSequences.from_rows(rows)


class TestScan:
    def test_max_supported_context_uses_int32_rolling_codes(self):
        sequence = np.full(11, 4, dtype=np.uint8)
        codes = np.empty(1, dtype=np.int32)

        ho_kmer_codes(sequence, 11, 0, 1, False, codes)

        assert codes.dtype == np.dtype(np.int32)
        assert int(codes[0]) == 5**11 - 1

    def test_custom_model_must_fill_both_finite_output_tracks(self, batch):
        class IncompleteModel(MotifModel):
            name = "incomplete"
            motif_length = 1

            def scan_into(self, sequence, forward, reverse, /):
                forward.fill(0.0)

        with pytest.raises(ModelInterfaceError, match="fill both output tracks"):
            scan(IncompleteModel(), batch, strands="both")

    def test_builtin_scan_rejects_float32_overflow(self):
        huge = np.finfo(np.float32).max
        model = PWM(
            "overflow",
            np.full((5, 2), huge, dtype=np.float32),
            (0.25, 0.25, 0.25, 0.25),
        )
        batch = EncodedSequences.from_rows([_seq(0, 0, 0)])

        with pytest.raises(ModelFormatError, match="non-finite"):
            scan(model, batch)

    def test_forward_matches_manual(self, pwm):
        seq = _seq(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)
        batch = EncodedSequences.from_rows([seq])
        result = scan(pwm, batch, strands="forward")
        n_pos = 20 - pwm.motif_length + 1
        assert len(result[0]) == n_pos
        expected = np.zeros(n_pos, dtype=np.float32)
        for pos in range(n_pos):
            total = 0.0
            for term in range(pwm.motif_length):
                total += float(pwm.weights[seq[pos + term], term])
            expected[pos] = total
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_reverse_complement_equivalence(self, pwm, batch):
        rev = scan(pwm, batch, strands="reverse")
        rc_batch = EncodedSequences.from_rows(
            [reverse_complement(batch[i]) for i in range(len(batch))]
        )
        fwd_rc = scan(pwm, rc_batch, strands="forward")
        for i in range(len(batch)):
            if len(rev[i]):
                assert np.allclose(rev[i], fwd_rc[i][::-1], atol=1e-6)

    def test_best_strand_is_max(self, pwm, batch):
        fwd = scan(pwm, batch, strands="forward")
        rev = scan(pwm, batch, strands="reverse")
        best = scan(pwm, batch, strands="best")
        for i in range(len(batch)):
            if len(fwd[i]):
                np.testing.assert_array_equal(best[i], np.maximum(fwd[i], rev[i]))

    def test_best_ties_keep_forward(self):
        m = PWM("m", np.zeros((5, 2), dtype=np.float32), (0.25, 0.25, 0.25, 0.25))
        seq = _seq(0, 1, 2)
        batch = EncodedSequences.from_rows([seq])
        best = scan(m, batch, strands="best")
        fwd = scan(m, batch, strands="forward")
        assert np.array_equal(best[0], fwd[0])

    def test_both_returns_pair(self, pwm, batch):
        pair = scan(pwm, batch, strands="both")
        fwd = scan(pwm, batch, strands="forward")
        rev = scan(pwm, batch, strands="reverse")
        for i in range(len(batch)):
            assert np.array_equal(pair.forward[i], fwd[i])
            assert np.array_equal(pair.reverse[i], rev[i])

    def test_short_sequence_empty(self, pwm):
        batch = EncodedSequences.from_rows([_seq(0, 1)])
        result = scan(pwm, batch, strands="forward")
        assert result[0].size == 0

    def test_empty_batch(self, pwm):
        batch = EncodedSequences(np.array([], dtype=np.uint8), np.array([0], dtype=np.int64))
        result = scan(pwm, batch, strands="forward")
        assert len(result) == 0

    def test_n_scoring(self):
        w = np.array(
            [[1.0, 2.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [-1.0, -1.0]],
            dtype=np.float32,
        )
        m = PWM("n", w, (0.25, 0.25, 0.25, 0.25))
        seq = _seq(4, 0, 1)
        batch = EncodedSequences.from_rows([seq])
        fwd = scan(m, batch, strands="forward")
        assert fwd[0].tolist() == [-1.0 + 2.0, 1.0 + 0.5]

    def test_invalid_strand_policy(self, pwm, batch):
        with pytest.raises(ValueError):
            scan(pwm, batch, strands="diagonal")
