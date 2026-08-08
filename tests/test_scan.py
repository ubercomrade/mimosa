import numpy as np
import pytest

from mimosa import BaMM, Dimont, PWM, SiteGA, Slim, EncodedSequences, pwm_from_pfm, scan
from mimosa.arrays import reverse_complement_batch
from mimosa.io.models import read_meme
from mimosa.scan import _scan_models_batch


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
        rc_batch = reverse_complement_batch(batch)
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

    def test_model_batch_dispatch_matches_serial(self, batch):
        models = [
            PWM("p2", np.zeros((5, 2), dtype=np.float32), (0.25,) * 4),
            PWM("p3", np.zeros((5, 3), dtype=np.float32), (0.25,) * 4),
            BaMM("b0", np.zeros((5, 2), dtype=np.float32), 0, 2),
            Dimont("d0", np.zeros((5, 2), dtype=np.float32), 0, 2),
            Slim("s0", np.zeros((5, 2), dtype=np.float32), 0, 2),
            SiteGA("sg", np.zeros((25, 3), dtype=np.float32), 3),
        ]
        packed = _scan_models_batch(models, batch)
        for index, model in enumerate(models):
            expected = scan(model, batch, strands="both")
            actual = packed.pair(index)
            np.testing.assert_array_equal(actual.forward.data, expected.forward.data)
            np.testing.assert_array_equal(actual.forward.offsets, expected.forward.offsets)
            np.testing.assert_array_equal(actual.reverse.data, expected.reverse.data)
            np.testing.assert_array_equal(actual.reverse.offsets, expected.reverse.offsets)
