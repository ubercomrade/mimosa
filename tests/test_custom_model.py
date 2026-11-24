"""A third-party custom MotifModel using only public imports (DoD requirement)."""

import numpy as np
import jax.numpy as jnp
import pytest

from mimosa import (
    BestPerSequence,
    MotifModel,
    compare,
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

    def score_sites(self, windows):
        weights = jnp.asarray(self._weights)
        return jnp.sum(weights[windows[:, 1:], jnp.arange(self.motif_length)], axis=1)


@pytest.fixture
def pwm():
    name, pfm = read_meme("examples/foxa2.meme")
    return pwm_from_pfm(pfm, name=name)


@pytest.fixture
def batch():
    return read_fasta("examples/foreground.fa")[0]


class TestCustomModel:
    @pytest.mark.parametrize("use_background", [False, True])
    @pytest.mark.parametrize("custom", [False, True])
    def test_monte_carlo_profile_matches_preparation(self, pwm, custom, use_background):
        import jax
        from mimosa import EncodedSequences, prepare_profile
        from mimosa.monte_carlo import _profile

        model = ShiftedPWM(pwm) if custom else pwm
        sequences = EncodedSequences.from_strings(["ACGTTGCA" * 5, "TTACG" * 6])
        background = EncodedSequences.from_strings(["GATTACA" * 7]) if use_background else None
        prepared = prepare_profile(model, sequences, background=background, circular=True)
        with jax.enable_x64(True):
            actual = np.asarray(_profile(model, sequences, background))
        np.testing.assert_array_equal(actual[0], prepared.bundle.forward.data)
        np.testing.assert_array_equal(actual[1], prepared.bundle.reverse.data)

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
                np.testing.assert_allclose(cus_rev[i], ref_rev[i][1 : 1 + n_cus], atol=1e-4)

    def test_compare_custom_vs_builtin(self, pwm, batch):
        custom = ShiftedPWM(pwm)
        result = compare(custom, pwm, batch, min_logerr=2.0)
        assert result.query == "shifted-" + pwm.name
        assert result.target == pwm.name
        # Asymmetric context is reversed on the minus strand, so its perfect
        # alignment is one position left of the plus-strand alignment.
        # Opposite orientations shift differently, changing position-wise calibration.
        assert float(result.score) == pytest.approx(1.0, abs=0.01)
        assert (result.orientation, result.offset) in (("++", 0), ("--", -1))

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
