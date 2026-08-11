import numpy as np
import pytest

from mimosa import (
    BestPerSequence,
    SiteCollection,
    ThresholdHits,
    TopFractionHits,
    build_pcm,
    extract_site_matrix,
    pcm_to_pfm,
    reconstruct_pfm,
    select_sites,
    site_strings,
)
from mimosa.io.fasta import read_fasta
from mimosa.io.models import read_meme
from mimosa.models import pwm_from_pfm


@pytest.fixture
def pwm():
    name, pfm = read_meme("examples/foxa2.meme")
    return pwm_from_pfm(pfm, name=name)


@pytest.fixture
def batch():
    return read_fasta("examples/foreground.fa")[0]


class TestSiteSelection:
    def test_best_per_sequence(self, pwm, batch):
        coll = select_sites(pwm, batch, BestPerSequence())
        assert isinstance(coll, SiteCollection)
        assert len(coll) == len(batch)
        assert np.all(coll.seq_indices == np.arange(len(batch)))

    def test_threshold_hits(self, pwm, batch):
        coll = select_sites(pwm, batch, ThresholdHits(5.0))
        assert np.all(coll.scores >= 5.0)

    def test_top_fraction(self, pwm, batch):
        coll = select_sites(pwm, batch, TopFractionHits(0.5, BestPerSequence()))
        assert len(coll) == len(batch) // 2

    def test_sorted_order(self, pwm, batch):
        coll = select_sites(pwm, batch, ThresholdHits(4.0))
        assert np.all(np.diff(coll.seq_indices) >= 0)
        for i in range(len(coll) - 1):
            if coll.seq_indices[i] == coll.seq_indices[i + 1]:
                assert coll.scores[i] >= coll.scores[i + 1]

    def test_reverse_sites_rc(self, pwm, batch):
        coll = select_sites(pwm, batch, BestPerSequence())
        sites = extract_site_matrix(batch, coll, pwm.motif_length)
        assert sites.shape == (pwm.motif_length, len(coll))
        strings = site_strings(sites)
        assert len(strings) == len(coll)
        assert all(len(s) == pwm.motif_length for s in strings)

    def test_strand_policies(self, pwm, batch):
        fwd = select_sites(pwm, batch, BestPerSequence(), strands="forward")
        rev = select_sites(pwm, batch, BestPerSequence(), strands="reverse")
        both = select_sites(pwm, batch, BestPerSequence(), strands="both")
        assert np.all(fwd.strands == 0)
        assert np.all(rev.strands == 1)
        assert len(both) == len(batch)

    def test_invalid_strand(self, pwm, batch):
        with pytest.raises(ValueError):
            select_sites(pwm, batch, BestPerSequence(), strands="sideways")


class TestSiteCollection:
    def test_validation(self):
        with pytest.raises(ValueError):
            SiteCollection(
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int8),
                np.array([1.0], dtype=np.float32),
        )


class TestPfmReconstruction:
    def test_reconstruct_matches_reference(self, pwm, batch):
        pfm = reconstruct_pfm(pwm, batch, BestPerSequence())
        assert pfm.shape == (4, pwm.motif_length)
        np.testing.assert_allclose(pfm[:, 0], [0.62623763, 0.02227723, 0.10148515, 0.25], atol=1e-6)

    def test_reconstruct_default_pseudocount(self, pwm, batch):
        pfm = reconstruct_pfm(pwm, batch, BestPerSequence())
        pfm2 = reconstruct_pfm(pwm, batch, BestPerSequence(), pseudocount=0.25)
        np.testing.assert_array_equal(pfm, pfm2)

    def test_no_sites_raises(self, pwm, batch):
        with pytest.raises(ValueError):
            reconstruct_pfm(pwm, batch, ThresholdHits(1e9))

    def test_pcm_to_pfm_zero_columns(self):
        pcm = np.zeros((4, 2), dtype=np.float32)
        with pytest.raises(Exception):
            pcm_to_pfm(pcm, pseudocount=0.0)
        pfm = pcm_to_pfm(pcm, pseudocount=0.25)
        np.testing.assert_allclose(pfm.sum(axis=0), 1.0)

    def test_build_pcm_skips_n(self):
        sites = np.array([[0, 4], [1, 2], [2, 1], [3, 0]], dtype=np.uint8)
        pcm = build_pcm(sites, 4)
        # site0 = ACGT (diagonal), site1 = NCGA: N skipped
        assert pcm[0, 0] == 1 and pcm[1, 1] == 1 and pcm[2, 2] == 1 and pcm[3, 3] == 1
        assert pcm[0, 3] == 1 and pcm[1, 2] == 1 and pcm[2, 1] == 1
        assert pcm[0, 1] == 0  # N not counted
