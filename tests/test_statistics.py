import numpy as np
import pytest

from mimosa import adjusted_pvalues, empirical_upper_tail_pvalue, evalue


def test_benjamini_hochberg_adjustment():
    np.testing.assert_allclose(adjusted_pvalues([0.01, 0.04, 0.03, 0.002]),
                               [0.02, 0.04, 0.04, 0.008])
    assert adjusted_pvalues([]).size == 0


@pytest.mark.parametrize("values", [[-0.1], [1.1], [np.nan]])
def test_bh_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        adjusted_pvalues(values)


def test_empirical_tail_includes_ties_and_pseudocount():
    assert empirical_upper_tail_pvalue([0.1, 0.5, 0.5, 0.9], 0.5) == 4 / 5
    assert empirical_upper_tail_pvalue([0.1, 0.5], 1) == 1 / 3


def test_evalue():
    assert evalue(0.1, 5) == 0.5
    with pytest.raises(ValueError):
        evalue(1.1, 5)
    with pytest.raises(ValueError):
        evalue(0.1, -1)
