import numpy as np
import pytest

from mimosa import (
    BaMM,
    MotifModel,
    PWM,
    EncodedSequences,
    SiteGA,
    pwm_from_pfm,
    reverse_complement,
    scan,
    site_start_offset,
    window_size,
)
from mimosa._kernels import ho_kmer_codes
from mimosa.errors import ModelFormatError, ModelInterfaceError
from mimosa.io.models import read_meme
from mimosa.scan import _scan_batch_into


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


class TestCircularScan:
    @staticmethod
    def _models():
        rng = np.random.default_rng(21)
        return [
            PWM("pwm", rng.normal(size=(5, 7)).astype(np.float32), (0.25,) * 4),
            BaMM("bamm", rng.normal(size=(125, 6)).astype(np.float32), 2, 6),
            SiteGA("sitega", rng.normal(size=(25, 6)).astype(np.float32), 6),
        ]

    @staticmethod
    def _sequences():
        rng = np.random.default_rng(22)
        # Includes N bases, an empty row, and rows shorter than every window.
        rows = [rng.integers(0, 5, size=n, dtype=np.uint8) for n in (40, 3, 0, 1, 17)]
        return EncodedSequences.from_rows(rows)

    @pytest.mark.parametrize("index", range(3))
    def test_matches_linear_scan_of_manually_wrapped_rows(self, index):
        model = self._models()[index]
        sequences = self._sequences()
        left = site_start_offset(model)
        right = window_size(model) - left - 1
        wrapped = EncodedSequences.from_rows(
            [
                row[np.arange(-left, len(row) + right) % len(row)] if len(row) else row
                for row in (sequences[i] for i in range(len(sequences)))
            ]
        )

        circular = _scan_batch_into(model, sequences, "both", circular=True)
        linear = scan(model, wrapped, strands="both")

        for strand, reference in (
            (circular.forward, linear.forward),
            (circular.reverse, linear.reverse),
        ):
            # Exactly one score per base, indexed by the site start.
            np.testing.assert_array_equal(strand.offsets, sequences.offsets)
            np.testing.assert_array_equal(strand.data, reference.data)

    def test_pwm_index_is_the_site_start_on_both_strands(self):
        model = self._models()[0]
        sequences = self._sequences()
        circular = _scan_batch_into(model, sequences, "both", circular=True)
        width = model.motif_length
        for row in range(len(sequences)):
            sequence = sequences[row]
            length = len(sequence)
            for start in range(length):
                site = sequence[(start + np.arange(width)) % length]
                forward = sum(model.weights[site[k], k] for k in range(width))
                reverse = sum(
                    model.weights[base if base == 4 else 3 - base, k]
                    for k, base in enumerate(site[::-1])
                )
                assert circular.forward[row][start] == pytest.approx(forward, abs=1e-4)
                assert circular.reverse[row][start] == pytest.approx(reverse, abs=1e-4)

    def test_context_model_reads_left_context_from_the_ring(self):
        rng = np.random.default_rng(23)
        pwm_weights = rng.normal(size=(5, 4)).astype(np.float32)
        bamm_weights = np.empty((25, 4), dtype=np.float32)
        for code in range(25):
            bamm_weights[code] = pwm_weights[code % 5]
        pwm = PWM("pwm", pwm_weights, (0.25,) * 4)
        bamm = BaMM("bamm", bamm_weights, 1, 4)
        sequences = self._sequences()

        from_pwm = _scan_batch_into(pwm, sequences, "both", circular=True)
        from_bamm = _scan_batch_into(bamm, sequences, "both", circular=True)
        np.testing.assert_allclose(from_bamm.forward.data, from_pwm.forward.data, atol=1e-5)
        np.testing.assert_allclose(from_bamm.reverse.data, from_pwm.reverse.data, atol=1e-5)

    def test_custom_model_uses_the_same_wrapping(self):
        class FirstBase(MotifModel):
            name = "first-base"
            motif_length = 2
            left_context = 1

            def scan_into(self, sequence, forward, reverse, /):
                for pos in range(forward.shape[0]):
                    forward[pos] = sequence[pos + 1]
                    reverse[pos] = sequence[pos]

        sequences = EncodedSequences.from_rows([_seq(0, 1, 2, 3), _seq(2)])
        circular = _scan_batch_into(FirstBase(), sequences, "both", circular=True)
        np.testing.assert_array_equal(circular.forward.data, [0, 1, 2, 3, 2])
        np.testing.assert_array_equal(circular.reverse.data, [3, 0, 1, 2, 2])

    def test_public_scan_stays_linear(self, pwm, batch):
        linear = scan(pwm, batch, strands="both")
        assert len(linear.forward[0]) == len(batch[0]) - pwm.motif_length + 1
        assert len(linear.forward[2]) == 0
