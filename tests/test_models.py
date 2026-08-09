import numpy as np
import pytest

from mimosa import (
    BaMM,
    Dimont,
    PWM,
    SiteGA,
    Slim,
    extend_pwm_with_n,
    pfm_to_pwm,
    pwm_from_pfm,
    site_start_offset,
    window_size,
)
from mimosa.errors import ModelDimensionError, ModelFormatError


def make_pwm(name="m", rows=5, cols=4):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(rows, cols)).astype(np.float32)
    return PWM(name, w, (0.25, 0.25, 0.25, 0.25))


class TestPWM:
    def test_construction(self):
        m = make_pwm()
        assert m.name == "m"
        assert m.motif_length == 4
        assert m.left_context == 0
        assert m.right_context == 0
        assert window_size(m) == 4
        assert site_start_offset(m) == 0

    def test_requires_5_rows(self):
        with pytest.raises(ModelDimensionError):
            PWM("x", np.zeros((4, 3), dtype=np.float32), (0.25, 0.25, 0.25, 0.25))

    def test_rejects_nonfinite(self):
        w = np.zeros((5, 3), dtype=np.float32)
        w[0, 0] = np.nan
        with pytest.raises(ModelFormatError):
            PWM("x", w, (0.25, 0.25, 0.25, 0.25))

    def test_rejects_non_two_dimensional_weights(self):
        with pytest.raises(ModelDimensionError):
            PWM("x", np.zeros((5, 3, 1), dtype=np.float32), (0.25, 0.25, 0.25, 0.25))

    def test_weights_do_not_alias_input(self):
        source = np.zeros((5, 3), dtype=np.float32)
        model = PWM("x", source, (0.25, 0.25, 0.25, 0.25))
        source[0, 0] = 1.0
        assert model.weights[0, 0] == 0.0

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            PWM("", np.zeros((5, 3), dtype=np.float32), (0.25, 0.25, 0.25, 0.25))

    def test_bad_background(self):
        with pytest.raises(ModelFormatError):
            PWM("x", np.zeros((5, 3), dtype=np.float32), (0.5, 0.25, 0.25, 0.0))

    def test_weights_readonly(self):
        m = make_pwm()
        with pytest.raises(ValueError):
            m.weights[0, 0] = 1.0


class TestPfmToPwm:
    def test_pfm_to_pwm(self):
        pfm = np.array([[0.7, 0.1], [0.1, 0.4], [0.1, 0.4], [0.1, 0.1]], dtype=np.float32)
        pwm = pfm_to_pwm(pfm)
        assert pwm.shape == (4, 2)
        assert np.all(np.isfinite(pwm))

    def test_requires_4_rows(self):
        with pytest.raises(ModelDimensionError):
            pfm_to_pwm(np.zeros((5, 2), dtype=np.float32))

    def test_rejects_bad_columns(self):
        pfm = np.array([[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        with pytest.raises(ModelFormatError):
            pfm_to_pwm(pfm)

    def test_extend_with_n(self):
        pwm4 = np.array([[1.0, 2.0], [0.0, 1.0], [-1.0, 0.0], [0.5, -1.0]], dtype=np.float32)
        pwm5 = extend_pwm_with_n(pwm4)
        assert pwm5.shape == (5, 2)
        assert pwm5[4].tolist() == [-1.0, -1.0]

    def test_pwm_from_pfm(self):
        pfm = np.array([[0.7, 0.1], [0.1, 0.4], [0.1, 0.4], [0.1, 0.1]], dtype=np.float32)
        m = pwm_from_pfm(pfm, name="x")
        assert isinstance(m, PWM)
        assert m.weights.shape == (5, 2)
        assert m.background == (0.25, 0.25, 0.25, 0.25)


class TestContextModels:
    def test_bamm(self):
        m = BaMM("b", np.zeros((25, 4), dtype=np.float32), 1, 4)
        assert m.left_context == 1
        assert m.right_context == 1
        assert window_size(m) == 6

    def test_bamm_row_validation(self):
        with pytest.raises(ModelDimensionError):
            BaMM("b", np.zeros((5, 4), dtype=np.float32), 1, 4)

    def test_context_representation_rejects_excessive_order(self):
        with pytest.raises(ModelDimensionError):
            Dimont("d", np.zeros((1, 1), dtype=np.float32), 11, 1)

    def test_dimont_limits_match_xml_reader(self):
        with pytest.raises(ModelDimensionError, match="must be <= 6"):
            Dimont("d", np.zeros((5**8, 1), dtype=np.float32), 7, 1)

    def test_dimont_and_slim(self):
        d = Dimont("d", np.zeros((625, 5), dtype=np.float32), 3, 5)
        s = Slim("s", np.zeros((625, 5), dtype=np.float32), 3, 5)
        assert d.left_context == 3
        assert s.left_context == 3

    def test_sitega(self):
        m = SiteGA("sg", np.zeros((25, 6), dtype=np.float32), 6)
        assert m.left_context == 0
        assert window_size(m) == 6

    def test_sitega_min_length(self):
        with pytest.raises(ModelDimensionError):
            SiteGA("sg", np.zeros((25, 1), dtype=np.float32), 1)
