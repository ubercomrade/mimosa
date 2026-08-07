"""A third-party custom MotifModel using only public imports (DoD requirement)."""

import numpy as np
import pytest

from mimosa import (
    BestPerSequence,
    MotifModel,
    PWM,
    compare,
    prepare_profile,
    reconstruct_pfm,
    select_sites,
)
from mimosa.io.fasta import read_fasta
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm


class ShiftedPWM(MotifModel):
    """Custom model: PWM shifted one base left; no private module access."""

    def __init__(self, pwm):
        self.pwm = pwm
        self._weights = pwm.weights

    @property
    def name(self):
        return "shifted-" + self.pwm.name

    @property
    def motif_length(self):
        return self.pwm.motif_length

    @property
    def left_context(self):
        return 1

    @property
    def right_context(self):
        return 0

    def scan_into(self, sequence, forward, reverse, /):
        import numpy as np

        w = self._weights
        n_pos = forward.shape[0]
        for pos in range(n_pos):
            total = 0.0
            for term in range(self.motif_length):
                total += float(w[sequence[pos + 1 + term], term])
            forward[pos] = total
        for pos in range(n_pos):
            total = 0.0
            for term in range(self.motif_length):
                base = sequence[pos + 1 + self.motif_length - 1 - term]
                if base == 4:
                    base = 4
                else:
                    base = 3 - base
                total += float(w[base, term])
            reverse[pos] = total


@pytest.fixture
def pwm():
    name, pfm = read_meme("examples/foxa2.meme")
    return pwm_from_pfm(pfm, name=name)


@pytest.fixture
def batch():
    return read_fasta("examples/foreground.fa")[0]


class TestCustomModel:
    def test_scan_parity_with_shifted_pwm(self, pwm, batch):
        custom = ShiftedPWM(pwm)
        # custom scan at p reads seq[p+1..p+11]; PWM scan of the N-padded
        # sequence reads padded[p..p+11] where padded[0]=N. They coincide.
        from mimosa import EncodedSequences, scan as scan_batch

        rows = [np.concatenate([[4], batch[i]]) for i in range(len(batch))]
        padded = EncodedSequences.from_rows(rows)
        ref_fwd = scan_batch(pwm, padded, strands="forward")
        ref_rev = scan_batch(pwm, padded, strands="reverse")
        cus_fwd = scan_batch(custom, batch, strands="forward")
        cus_rev = scan_batch(custom, batch, strands="reverse")
        for i in range(len(batch)):
            L = len(batch[i])
            if L >= custom.motif_length + 1:
                n_cus = L - custom.motif_length - 1 + 1  # window = 12
                assert len(cus_fwd[i]) == n_cus
                # padded[q..q+10] == batch[q-1..q+9]; custom[p] == batch[p+1..p+11]
                # -> q = p + 2
                np.testing.assert_allclose(cus_fwd[i], ref_fwd[i][2 : 2 + n_cus], atol=1e-4)
                np.testing.assert_allclose(cus_rev[i], ref_rev[i][2 : 2 + n_cus], atol=1e-4)

    def test_compare_custom_vs_builtin(self, pwm, batch):
        custom = ShiftedPWM(pwm)
        result = compare(custom, pwm, batch)
        assert result.query == "shifted-" + pwm.name
        assert result.target == pwm.name
        assert 0 <= float(result.score) <= 1

    def test_sites_via_custom(self, pwm, batch):
        custom = ShiftedPWM(pwm)
        coll = select_sites(custom, batch, BestPerSequence())
        assert len(coll) == len(batch)

    def test_reconstruct_pfm_via_custom(self, pwm, batch):
        custom = ShiftedPWM(pwm)
        pfm = reconstruct_pfm(custom, batch, BestPerSequence())
        assert pfm.shape == (4, custom.motif_length)
        np.testing.assert_allclose(pfm.sum(axis=0), 1.0, atol=1e-5)

    def test_fingerprint_none_by_default(self, pwm):
        custom = ShiftedPWM(pwm)
        assert custom.fingerprint() is None
