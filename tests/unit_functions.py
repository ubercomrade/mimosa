# ruff: noqa: F403,F405

from tests.unit_support import *


def _reference_shift_score(
    scores1,
    lengths1,
    scores2,
    lengths2,
    query_rows,
    query_positions,
    target_rows,
    target_positions,
    shift,
    radius,
    realign_window,
    metric,
):
    """Small allocation-heavy profile alignment reference kept only in tests."""
    candidates = set()
    for row, position1 in zip(query_rows, query_positions, strict=True):
        position2 = int(position1) + shift
        if (
            int(position1) - radius >= 0
            and int(position1) + radius < int(lengths1[row])
            and position2 - radius >= 0
            and position2 + radius < int(lengths2[row])
        ):
            candidates.add((int(row), int(position1)))
    for row, position2 in zip(target_rows, target_positions, strict=True):
        expected = int(position2) - shift
        left = max(0, expected - realign_window)
        right = min(int(lengths1[row]) - 1, expected + realign_window)
        if left > right:
            continue
        position1 = left + int(np.argmax(scores1[row, left : right + 1]))
        aligned2 = position1 + shift
        if (
            position1 - radius >= 0
            and position1 + radius < int(lengths1[row])
            and aligned2 - radius >= 0
            and aligned2 + radius < int(lengths2[row])
        ):
            candidates.add((int(row), position1))

    ordered = sorted(candidates)
    if not ordered:
        return 0.0, 0
    windows1 = np.array(
        [scores1[row, position - radius : position + radius + 1] for row, position in ordered], dtype=np.float32
    )
    windows2 = np.array(
        [scores2[row, position + shift - radius : position + shift + radius + 1] for row, position in ordered],
        dtype=np.float32,
    )
    if metric == "co":
        score = calc_co(windows1, windows2)
    elif metric == "dice":
        score = calc_dice(windows1, windows2)
    else:
        functions = {"co_rowwise": rowwise_co, "dice_rowwise": rowwise_dice, "cosine": rowwise_cosine}
        values = functions[metric](windows1, windows2)
        score = float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 0.0
    return score, len(ordered)


@pytest.mark.parametrize("metric", ["co", "dice", "co_rowwise", "dice_rowwise", "cosine"])
@pytest.mark.parametrize("shift", [-1, 0, 1])
def test_fused_profile_alignment_matches_reference(metric, shift):
    """Fused serial and parallel kernels must preserve selection and metric semantics."""
    from mimosa.functions.alignment import build_anchor_csr, make_alignment_workspace, score_shift

    scores1 = np.array(
        [[0.0, 2.0, 1.0, 3.0, 0.0], [1.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    scores2 = np.array(
        [[0.0, 1.0, 2.0, 2.0, 0.0], [0.5, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    lengths1 = np.array([5, 3, 0], dtype=np.int32)
    lengths2 = np.array([5, 4, 0], dtype=np.int32)
    query_rows = np.array([0, 0, 1], dtype=np.int32)
    query_positions = np.array([1, 3, 2], dtype=np.int32)
    target_rows = np.array([0, 0, 1, 1], dtype=np.int32)
    target_positions = np.array([1, 2, 1, 1], dtype=np.int32)
    query_csr = build_anchor_csr(query_rows, query_positions, 3)
    target_csr = build_anchor_csr(target_rows, target_positions, 3)
    expected_score, expected_sites = _reference_shift_score(
        scores1,
        lengths1,
        scores2,
        lengths2,
        query_rows,
        query_positions,
        target_rows,
        target_positions,
        shift,
        0,
        1,
        metric,
    )

    observed = []
    for use_parallel in (False, True):
        workspace = make_alignment_workspace(3, 5)
        observed.append(
            score_shift(
                scores1,
                lengths1,
                scores2,
                lengths2,
                query_csr,
                target_csr,
                shift,
                0,
                1,
                metric,
                workspace,
                1,
                use_parallel,
            )
        )

    for score, n_sites in observed:
        assert n_sites == expected_sites
        np.testing.assert_allclose(score, expected_score, rtol=1e-6, atol=1e-7)


def test_pfm_to_pwm_basic():
    """Test basic PFM to PWM conversion"""
    # Create a simple PFM with uniform values
    pfm = np.array([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]])

    pwm = pfm_to_pwm(pfm)

    # Verify shape is preserved
    assert pwm.shape == pfm.shape

    # Verify that PWM values are log ratios
    expected = np.log((pfm + 0.0001) / 0.25)
    np.testing.assert_allclose(pwm, expected, rtol=1e-6)


def test_pfm_to_pwm_with_zeros():
    """Test PFM to PWM conversion with zeros"""
    pfm = np.array([[0.0, 0.5], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

    pwm = pfm_to_pwm(pfm)

    # Verify shape is preserved
    assert pwm.shape == pfm.shape

    # Verify that PWM values are calculated correctly even with zeros
    expected = np.log((pfm + 0.0001) / 0.25)
    np.testing.assert_allclose(pwm, expected, rtol=1e-6)


def test_pcm_to_pfm_basic():
    """Test basic PCM to PFM conversion"""
    pcm = np.array([[2, 3], [1, 1], [1, 0], [0, 0]], dtype=float)

    pfm = pcm_to_pfm(pcm)

    # Verify shape is preserved
    assert pfm.shape == pcm.shape

    # Calculate expected manually
    number_of_sites = pcm.sum(axis=0)
    expected = (pcm + 0.25) / (number_of_sites + 1)
    np.testing.assert_allclose(pfm, expected, rtol=1e-6)


def test_score_seq_basic():
    """Test basic sequence scoring"""
    # Simple scoring model
    model = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    # DNA sequence as numerical representation [A, C, G, T] -> [0, 1, 2, 3]
    num_site = np.array([0, 1, 2], dtype=np.int8)
    kmer = 1

    score = score_seq(num_site, kmer, model)

    # With kmer=1, the function should compute the sum of model values at positions
    expected_score = 1.0 + 2.0 + 3.0
    assert score == expected_score


def test_precision_recall_curve_basic():
    """Test basic precision-recall curve calculation"""
    classification = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    precision, recall, thresholds = precision_recall_curve(classification, scores)

    # Verify shapes are consistent
    assert precision.shape == recall.shape == thresholds.shape
    # Verify bounds
    assert np.all(precision >= 0) and np.all(precision <= 1.1)  # Allow for small numerical errors
    assert np.all(recall >= 0) and np.all(recall <= 1.1)


def test_roc_curve_basic():
    """Test basic ROC curve calculation"""
    classification = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    tpr, fpr, thresholds = roc_curve(classification, scores)

    # Verify shapes are consistent
    assert tpr.shape == fpr.shape == thresholds.shape
    # Verify bounds
    assert np.all(tpr >= 0) and np.all(tpr <= 1.1)
    assert np.all(fpr >= 0) and np.all(fpr <= 1.1)


def test_cut_roc_basic():
    """Test basic ROC curve cutting"""
    tpr = np.array([0.0, 0.5, 1.0])
    fpr = np.array([0.0, 0.2, 0.5])
    thr = np.array([np.inf, 0.5, 0.0])
    score_cutoff = 0.6

    tpr_cut, fpr_cut, thr_cut = cut_roc(tpr, fpr, thr, score_cutoff)

    # Verify that the function returns arrays
    assert isinstance(tpr_cut, np.ndarray)
    assert isinstance(fpr_cut, np.ndarray)
    assert isinstance(thr_cut, np.ndarray)


def test_cut_prc_basic():
    """Test basic PRC curve cutting"""
    rec = np.array([0.0, 0.5, 1.0])
    prec = np.array([1.0, 0.8, 0.6])
    thr = np.array([np.inf, 0.5, 0.0])
    score_cutoff = 0.6

    rec_cut, prec_cut, thr_cut = cut_prc(rec, prec, thr, score_cutoff)

    # Verify that the function returns arrays
    assert isinstance(rec_cut, np.ndarray)
    assert isinstance(prec_cut, np.ndarray)
    assert isinstance(thr_cut, np.ndarray)


def test_standardized_pauc_basic():
    """Test basic standardized partial AUC calculation"""
    pauc_raw = 0.7
    pauc_min = 0.5
    pauc_max = 1.0

    standardized = standardized_pauc(pauc_raw, pauc_min, pauc_max)

    # Manual calculation: 0.5 * (1.0 + (0.7 - 0.5) / (1.0 - 0.5))
    expected = 0.5 * (1.0 + (0.7 - 0.5) / (1.0 - 0.5))
    assert abs(standardized - expected) < 1e-6


def test_scores_to_empirical_log_tail_basic():
    """Empirical score normalization should preserve dense masked layout."""
    data = np.array([1.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float32)
    offsets = np.array([0, 2, 4, 5], dtype=np.int64)
    score_batch = _score_batch_from_flat(data, offsets)

    transformed = scores_to_empirical_log_tail(score_batch)

    assert transformed["values"].shape == score_batch["values"].shape
    assert transformed["mask"].shape == score_batch["mask"].shape
    np.testing.assert_array_equal(transformed["lengths"], score_batch["lengths"])


def test_scores_to_empirical_log_tail_bundle_matches_table_apply_with_empty_rows():
    """Fused two-strand normalization must preserve the old table-apply oracle."""
    plus = make_score_batch([np.array([3.0, -0.0, 1.0], dtype=np.float32), np.array([0.25], dtype=np.float32)])
    minus = make_score_batch([np.array([2.0, 1.0, 2.0], dtype=np.float32), np.array([0.5], dtype=np.float32)])
    bundle = make_strand_bundle(plus, minus)
    table = build_score_log_tail_table(flatten_profile_bundle(bundle))
    expected = apply_score_log_tail_table_to_profile_bundle(bundle, table)
    actual = scores_to_empirical_log_tail_bundle(bundle)

    np.testing.assert_array_equal(actual["values"], expected["values"])
    np.testing.assert_array_equal(actual["lengths"], expected["lengths"])


def test_scores_to_empirical_log_tail_bundle_preserves_tied_full_rows():
    """Unstable sorting must preserve exact group ranks for tied dense scores."""
    plus = make_score_batch(
        [
            np.array([2.0, 1.0, 2.0, -0.0], dtype=np.float32),
            np.array([1.0, 3.0, 1.0, 0.0], dtype=np.float32),
        ]
    )
    minus = make_score_batch(
        [
            np.array([1.0, 3.0, 2.0, 0.0], dtype=np.float32),
            np.array([2.0, 1.0, 3.0, -0.0], dtype=np.float32),
        ]
    )
    bundle = make_strand_bundle(plus, minus)
    table = build_score_log_tail_table(flatten_profile_bundle(bundle))
    expected = apply_score_log_tail_table_to_profile_bundle(bundle, table)

    actual = scores_to_empirical_log_tail_bundle(bundle)

    np.testing.assert_array_equal(actual["values"], expected["values"])
    np.testing.assert_array_equal(actual["lengths"], expected["lengths"])


def test_build_score_log_tail_table_returns_float32():
    """Score log-tail tables should use float32 for faster downstream lookup."""
    table = build_score_log_tail_table(np.array([0.1, 0.3, 0.2, 0.3], dtype=np.float64))
    assert table.dtype == np.float32


def test_apply_score_log_tail_table_preserves_padding_for_empty_rows():
    """Lookup normalization should keep padding intact when no valid scores are present."""
    score_batch = make_score_batch([np.array([], dtype=np.float32)])
    table = build_score_log_tail_table(np.array([0.1, 0.3, 0.2], dtype=np.float32))

    transformed = apply_score_log_tail_table(score_batch, table)

    np.testing.assert_array_equal(transformed["mask"], score_batch["mask"])
    np.testing.assert_array_equal(transformed["values"], score_batch["values"])
