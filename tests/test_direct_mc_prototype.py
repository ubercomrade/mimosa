import numpy as np
import pytest

from benchmarks.direct_mc_prototype import direct_mc, permutations
from mimosa import EncodedSequences, read_model


def test_pair_is_independent_of_collection_and_counts_ties(monkeypatch):
    a = read_model("examples/foxa2.meme")
    b = read_model("examples/gata2.meme")
    dna = EncodedSequences.from_rows(
        np.random.default_rng(9).integers(0, 4, (12, 60), dtype=np.uint8)
    )
    monkeypatch.setattr("benchmarks.direct_mc_prototype.permutations",
                        lambda model, *args: np.tile(np.arange(4), (model.motif_length, 1)))
    with pytest.warns(UserWarning):
        one = direct_mc([a], [b], dna, k=2, seed=7, min_logerr=2, search_range=1)
    with pytest.warns(UserWarning):
        matrix = direct_mc([b, a], [a, b], dna, k=2, seed=7, min_logerr=2, search_range=1)
    assert one[0][0, 0] == matrix[0][1, 1]
    assert one[1][0, 0] == matrix[1][1, 1] == 2
    assert one[2][0, 0] == matrix[2][1, 1] == 1


@pytest.mark.parametrize("group", ["full", "gc"])
def test_permutation_stream(group):
    model = read_model("examples/foxa2.meme")
    a = permutations(model, 42, 3, group)
    np.testing.assert_array_equal(a, permutations(model, 42, 3, group))
    np.testing.assert_array_equal(np.sort(a, axis=1), np.tile(np.arange(4), (len(a), 1)))
    if group == "gc":
        assert np.all(np.isin(a[:, [0, 3]], [0, 3]))
        assert np.all(np.isin(a[:, [1, 2]], [1, 2]))
