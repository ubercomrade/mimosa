import os

import numpy as np
import pytest

from mimosa import PWM, BaMM, SiteGA, Slim, Dimont, read_model
from mimosa.errors import InvariantError, ModelFormatError
from mimosa.io.bundles import (
    content_fingerprint_float32,
    content_fingerprint_float64,
    model_fingerprint,
    read_model_bundle,
    read_null_bundle,
    write_model,
    write_null_bundle,
)
from mimosa.io.fasta import read_fasta, read_scores
from mimosa.io.models import read_bamm, read_dimont, read_meme, read_pfm, read_sitega, read_slim
from mimosa.models import pwm_from_pfm
from mimosa.statistics import NullDistribution, build_null


class TestFasta:
    def test_read_fasta(self):
        batch, names = read_fasta("examples/foreground.fa")
        assert len(batch) == len(names) == 100
        assert all(len(n) > 0 for n in names)

    def test_read_fasta_encoding(self):
        batch, _ = read_fasta("examples/foreground.fa")
        assert np.all(batch.data <= 4)

    def test_read_fasta_empty(self, tmp_path):
        p = tmp_path / "empty.fa"
        p.write_text("")
        with pytest.raises(ModelFormatError):
            read_fasta(str(p))

    def test_read_fasta_header_only(self, tmp_path):
        p = tmp_path / "x.fa"
        p.write_text(">only_header\n")
        batch, names = read_fasta(str(p))
        assert len(batch) == 1
        assert batch[0].size == 0


class TestScores:
    def test_read_scores(self):
        s = read_scores("examples/scores_1.fasta")
        assert s.name == "scores_1"
        assert len(s) == 1000
        assert np.all(np.isfinite(s.scores.data))

    def test_read_scores_comma_separated(self, tmp_path):
        p = tmp_path / "s.fasta"
        p.write_text(">a\n1.0, 2.0, 3.0\n>b\n4.0\n")
        s = read_scores(str(p))
        assert len(s) == 2
        assert s.scores[0].tolist() == [1.0, 2.0, 3.0]
        assert s.scores[1].tolist() == [4.0]

    def test_read_scores_missing_header(self, tmp_path):
        p = tmp_path / "s.fasta"
        p.write_text("1.0\n")
        with pytest.raises(ModelFormatError):
            read_scores(str(p))

    def test_read_scores_invalid(self, tmp_path):
        p = tmp_path / "s.fasta"
        p.write_text(">a\n1.0 xyz\n")
        with pytest.raises(ModelFormatError):
            read_scores(str(p))


class TestPwmReaders:
    def test_public_read_model(self):
        model = read_model("examples/foxa2.meme")
        assert isinstance(model, PWM)
        assert model.name == "MA0047.3"

    def test_read_meme(self):
        name, pfm = read_meme("examples/foxa2.meme")
        assert name == "MA0047.3"
        assert pfm.shape == (4, 11)
        np.testing.assert_allclose(pfm.sum(axis=0), 1.0, atol=1e-4)

    def test_read_meme_index(self):
        _, pfm0 = read_meme("examples/foxa2.meme", index=0)
        with pytest.raises(ModelFormatError):
            read_meme("examples/foxa2.meme", index=5)

    def test_read_pfm(self):
        name, pfm = read_pfm("examples/pif4.pfm")
        assert pfm.shape[0] == 4
        np.testing.assert_allclose(pfm.sum(axis=0), 1.0, atol=1e-4)

    def test_read_meme_validation(self, tmp_path):
        p = tmp_path / "bad.meme"
        p.write_text("MOTIF x\nletter-probability matrix: alength= 4 w= 3\n0.5 0.5 0.5 0.5\n0.5 0.5 0.5 0.5\n")
        with pytest.raises(ModelFormatError):
            read_meme(str(p))

    def test_read_meme_no_motifs(self, tmp_path):
        p = tmp_path / "none.meme"
        p.write_text("header\n")
        with pytest.raises(ModelFormatError):
            read_meme(str(p))

    def test_custom_reader_and_ambiguity(self, tmp_path):
        _, pfm = read_meme("examples/foxa2.meme")
        model = pwm_from_pfm(pfm, name="custom")
        path = tmp_path / "model.custom"
        path.write_text("ignored")

        class Reader:
            formats = ("custom",)

            def __init__(self, selected):
                self.selected = selected

            def probe(self, candidate):
                return self.selected

            def read(self, candidate, **kwargs):
                return model

        assert read_model(path, readers=(Reader(True),)).name == "custom"
        assert read_model(path, readers=(Reader(False), Reader(True))).name == "custom"
        with pytest.raises(ModelFormatError):
            read_model(path, format="custom", readers=(Reader(True), Reader(True)))


class TestBammReader:
    @pytest.fixture
    def bamm_path(self, tmp_path):
        p = tmp_path / "myog.ihbcp"
        with open(p, "w") as f:
            for position in range(1, 15):
                for order in range(5):
                    values = ["0.1", "0.2", "0.3", "0.4"] * (4**order)
                    f.write(" ".join(values) + "\n")
                if position < 14:
                    f.write("\n")
        return str(p)

    def test_read_bamm(self, bamm_path):
        m = read_bamm(bamm_path)
        assert m.name == "myog"
        assert m.motif_length == 14
        assert m.order == 4
        assert m.weights.shape == (5**5, 14)

    def test_read_bamm_order0(self, bamm_path):
        m = read_bamm(bamm_path, order=0)
        assert m.order == 0
        assert m.weights.shape == (5, 14)

    def test_read_bamm_clamped(self, bamm_path):
        m = read_bamm(bamm_path, order=100)
        assert m.order == 4

    @pytest.mark.parametrize(
        "path",
        [
            "examples/foxa2.ihbcp",
            "examples/gata2.ihbcp",
            "examples/gata4.ihbcp",
            "examples/myog.ihbcp",
        ],
    )
    def test_read_official_bamm_examples(self, path):
        model = read_bamm(path)
        assert model.order == 4
        assert model.motif_length > 0
        assert np.all(np.isfinite(model.weights))

    def test_read_bamm_bad_range(self, tmp_path):
        p = tmp_path / "bad.ihbcp"
        p.write_text("0.25 0.25 -0.1 0.85\n")
        with pytest.raises(ModelFormatError):
            read_bamm(str(p))

    def test_read_bamm_bad_width(self, tmp_path):
        p = tmp_path / "bad.ihbcp"
        p.write_text("0.2 0.2 0.2\n")
        with pytest.raises(ModelFormatError):
            read_bamm(str(p))


class TestSitega:
    def test_read(self):
        m = read_sitega("tests/fixtures/sitega.mat")
        assert m.name == "Bootatrap"
        assert m.motif_length == 12
        assert m.weights.shape == (25, 12)

    def test_malformed(self, tmp_path):
        p = tmp_path / "bad.mat"
        p.write_text("name\n0\tLPD\n")
        with pytest.raises(ModelFormatError):
            read_sitega(str(p))


class TestXmlReaders:
    def test_read_dimont(self):
        m = read_dimont("tests/fixtures/stat_dimont-model-1.xml")
        assert m.order == 3
        assert m.motif_length == 5
        assert m.weights.shape == (625, 5)

    def test_read_slim(self):
        m = read_slim("tests/fixtures/slim/example-model-1.xml")
        assert m.order == 5
        assert m.motif_length == 15
        assert m.weights.shape == (15625, 15)

    def test_dimont_wrong_root(self, tmp_path):
        p = tmp_path / "d.xml"
        p.write_text("<notdimont/>")
        with pytest.raises(ModelFormatError):
            read_dimont(str(p))

    def test_malformed_xml_is_typed(self, tmp_path):
        p = tmp_path / "d.xml"
        p.write_text("<not-valid>")
        with pytest.raises(ModelFormatError):
            read_dimont(str(p))


class TestFingerprints:
    def test_float32_reference(self):
        # verified against Julia Mimosa.content_fingerprint
        arr = np.array([0.5, 1.5], dtype=np.float32)
        assert (
            content_fingerprint_float32(arr)
            == "aa59046c133d7e1797fc98d7d409d319dcf51e64d56218477d60a25f05839fe2"
        )

    def test_float64_reference(self):
        # verified against Julia Mimosa.content_fingerprint
        arr = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        assert (
            content_fingerprint_float64(arr)
            == "008a3a430630b9eda48b5288efbf3468022c1b774291f774ace3791d40866cc7"
        )

    def test_matrix_uses_julia_shape_and_order(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        assert (
            content_fingerprint_float32(arr)
            == "6d71d1b6219c51609d1816faa0e172a88462b37e25ed350e782c013eabc7aa59"
        )

    def test_model_fingerprint_stable(self):
        name, pfm = read_meme("examples/foxa2.meme")
        m = pwm_from_pfm(pfm, name=name)
        assert model_fingerprint(m) == model_fingerprint(m)
        assert len(model_fingerprint(m)) == 64

    def test_model_fingerprint_changes_with_weights(self):
        name, pfm = read_meme("examples/foxa2.meme")
        m1 = pwm_from_pfm(pfm, name=name)
        m2 = pwm_from_pfm(pfm, name=name)
        assert model_fingerprint(m1) == model_fingerprint(m2)
        # perturb a weight directly through a rebuilt PWM
        w = m1.weights.copy()
        w[0, 0] += 0.5
        m3 = PWM(m1.name, w, m1.background)
        assert model_fingerprint(m1) != model_fingerprint(m3)


class TestModelBundle:
    def test_roundtrip_pwm(self, tmp_path):
        name, pfm = read_meme("examples/foxa2.meme")
        m = pwm_from_pfm(pfm, name=name)
        path = str(tmp_path / "m")
        write_model(path, m)
        m2 = read_model_bundle(path)
        assert isinstance(m2, PWM)
        assert m2.name == m.name
        assert m2.background == m.background
        np.testing.assert_array_equal(m2.weights, m.weights)

    def test_roundtrip_bamm(self, tmp_path):
        m = read_bamm("/tmp/opencode/myog.ihbcp") if os.path.exists("/tmp/opencode/myog.ihbcp") else self._make_bamm(tmp_path)
        path = str(tmp_path / "b")
        write_model(path, m)
        m2 = read_model_bundle(path)
        assert isinstance(m2, BaMM)
        assert m2.order == m.order
        np.testing.assert_array_equal(m2.weights, m.weights)

    @staticmethod
    def _make_bamm(tmp_path):
        p = tmp_path / "myog.ihbcp"
        with open(p, "w") as f:
            for position in range(1, 15):
                for order in range(5):
                    values = ["0.1", "0.2", "0.3", "0.4"] * (4**order)
                    f.write(" ".join(values) + "\n")
                if position < 14:
                    f.write("\n")
        return read_bamm(str(p))

    def test_roundtrip_sitega(self, tmp_path):
        m = read_sitega("tests/fixtures/sitega.mat")
        path = str(tmp_path / "s")
        write_model(path, m)
        m2 = read_model_bundle(path)
        assert isinstance(m2, SiteGA)
        np.testing.assert_array_equal(m2.weights, m.weights)

    def test_roundtrip_dimont(self, tmp_path):
        m = read_dimont("tests/fixtures/stat_dimont-model-1.xml")
        path = str(tmp_path / "d")
        write_model(path, m)
        m2 = read_model_bundle(path)
        assert isinstance(m2, Dimont)
        assert m2.order == m.order
        np.testing.assert_array_equal(m2.weights, m.weights)

    def test_roundtrip_slim(self, tmp_path):
        m = read_slim("tests/fixtures/slim/example-model-1.xml")
        path = str(tmp_path / "sl")
        write_model(path, m)
        m2 = read_model_bundle(path)
        assert isinstance(m2, Slim)
        np.testing.assert_array_equal(m2.weights, m.weights)

    def test_refuses_overwrite(self, tmp_path):
        name, pfm = read_meme("examples/foxa2.meme")
        m = pwm_from_pfm(pfm, name=name)
        path = str(tmp_path / "m")
        write_model(path, m)
        with pytest.raises(Exception):
            write_model(path, m)

    def test_rejects_custom_model(self, tmp_path):
        class Custom:
            pass

        with pytest.raises(InvariantError):
            write_model(str(tmp_path / "custom"), Custom())

    def test_rejects_corrupt_checksum(self, tmp_path):
        name, pfm = read_meme("examples/foxa2.meme")
        m = pwm_from_pfm(pfm, name=name)
        path = str(tmp_path / "m")
        write_model(path, m)
        blob = os.path.join(path, "data", "weights.bin")
        with open(blob, "r+b") as f:
            f.seek(0)
            f.write(bytes([f.read(1)[0] ^ 0xFF]))
        with pytest.raises(ModelFormatError):
            read_model_bundle(path)

    def test_rejects_wrong_version(self, tmp_path):
        name, pfm = read_meme("examples/foxa2.meme")
        m = pwm_from_pfm(pfm, name=name)
        path = str(tmp_path / "m")
        write_model(path, m)
        manifest = os.path.join(path, "manifest.toml")
        with open(manifest) as f:
            content = f.read()
        with open(manifest, "w") as f:
            f.write(content.replace("format_version = 2", "format_version = 1"))
        with pytest.raises(ModelFormatError):
            read_model_bundle(path)


class TestNullBundle:
    @pytest.fixture
    def dist(self):
        models = []
        for f in ["foxa2.meme", "gata2.meme", "gata4.meme"]:
            name, pfm = read_meme(f"examples/{f}")
            models.append(pwm_from_pfm(pfm, name=name))
        batch, _ = read_fasta("examples/foreground.fa")
        return build_null(models, sequences=batch, n_samples=20, seed=7)

    def test_roundtrip(self, tmp_path, dist):
        path = str(tmp_path / "null")
        write_null_bundle(path, dist)
        d = read_null_bundle(path)
        assert d["n_null"] == 20
        assert d["metric"] == "co"
        np.testing.assert_array_equal(d["raw_scores"], dist.raw_scores)
        assert d["pairs"][0][0] == dist.pairs[0][0]
        assert d["contract"]["raw_scores_fingerprint"] == dist.contract["raw_scores_fingerprint"]

    def test_rejects_previous_format_version(self, tmp_path, dist):
        path = str(tmp_path / "null")
        write_null_bundle(path, dist)
        manifest = os.path.join(path, "manifest.toml")
        with open(manifest) as f:
            content = f.read()
        with open(manifest, "w") as f:
            f.write(content.replace("format_version = 8", "format_version = 7"))
        with pytest.raises(ModelFormatError):
            read_null_bundle(path)

    def test_rejects_removed_metric(self, tmp_path, dist):
        path = str(tmp_path / "null")
        write_null_bundle(path, dist)
        manifest = os.path.join(path, "manifest.toml")
        with open(manifest) as f:
            content = f.read()
        with open(manifest, "w") as f:
            f.write(content.replace('metric = "co"', 'metric = "co_rowwise"'))
        with pytest.raises(ModelFormatError):
            read_null_bundle(path)

    def test_raw_fingerprint_validated(self, tmp_path, dist):
        path = str(tmp_path / "null")
        write_null_bundle(path, dist)
        scores_path = os.path.join(path, "data", "raw_null_scores.npy")
        # corrupt a payload byte -> checksum mismatch
        with open(scores_path, "r+b") as f:
            f.seek(-8, 2)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 0xFF]))
        with pytest.raises(ModelFormatError):
            read_null_bundle(path)

    def test_rejects_empty_null_distribution(self, tmp_path, dist):
        path = str(tmp_path / "null")
        write_null_bundle(path, dist)
        manifest = os.path.join(path, "manifest.toml")
        with open(manifest) as f:
            content = f.read()
        with open(manifest, "w") as f:
            f.write(content.replace("n_null = 20", "n_null = 0"))
        with pytest.raises(ModelFormatError, match="n_null"):
            read_null_bundle(path)

    def test_rejects_non_finite(self, tmp_path, dist):
        dist2 = NullDistribution(
            strategy=dist.strategy,
            metric=dist.metric,
            raw_scores=np.array([1.0, np.nan]),
            pairs=dist.pairs[:2],
            n_null=2,
            n_models=dist.n_models,
            model_type=dist.model_type,
            shuffle=dist.shuffle,
            seed=dist.seed,
            sampling_version=dist.sampling_version,
            model_collection_fingerprint=dist.model_collection_fingerprint,
            sequence_fingerprint=dist.sequence_fingerprint,
            background_fingerprint=dist.background_fingerprint,
            contract=dist.contract,
        )
        with pytest.raises(Exception):
            write_null_bundle(str(tmp_path / "bad"), dist2)
