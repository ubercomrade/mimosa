# ruff: noqa: F403,F405

from tests.unit_support import *


def test_create_comparator_config():
    """Test ComparatorConfig creation and factory function"""
    # Test factory function with defaults
    config = create_comparator_config()
    assert config["metric"] == "pcc"
    assert config["seed"] is None
    assert config["pfm_top_fraction"] == pytest.approx(0.05)
    assert config["profile_normalization"] == "empirical_log_tail"
    assert config["n_jobs"] is None
    assert config["pvalue"] is False
    assert "n_permutations" not in config

    # Test factory function with custom parameters
    config = create_comparator_config(metric="co", seed=42, pfm_top_fraction=0.2, pvalue=True)
    assert config["metric"] == "co"
    assert config["seed"] == 42
    assert config["pfm_top_fraction"] == pytest.approx(0.2)
    assert config["pvalue"] is True


def test_create_comparator_config_resolves_n_jobs():
    """Explicit n_jobs should drive the effective parallel setting."""
    config = create_comparator_config(n_jobs=4)
    assert config["n_jobs"] == 4


@pytest.mark.parametrize("kwargs", [{"n_jobs": 0}, {"n_jobs": -2}])
def test_create_comparator_config_validates_thread_counts(kwargs):
    """Thread-count settings should accept only positive values or -1."""
    with pytest.raises(ValueError, match="positive or -1"):
        create_comparator_config(**kwargs)


def test_compare_passes_n_jobs_to_strategy(monkeypatch):
    """Execution boundary should pass the validated config to the strategy."""
    observed = []

    def fake_strategy(model1, model2, sequences, cfg):
        observed.append(cfg["n_jobs"])
        return {"score": 1.0}

    monkeypatch.setitem(comparison_registry, "thread_test", fake_strategy)

    result = compare(
        model1=None,
        model2=None,
        strategy="thread_test",
        config=create_comparator_config(n_jobs=1),
    )

    assert result == {"score": 1.0}
    assert observed == [1]


def test_cli_maps_jobs_to_n_jobs_for_profile_config():
    """CLI --jobs should reach comparator config as n_jobs."""
    args = SimpleNamespace(
        mode="profile",
        metric="co",
        jobs=3,
        seed=5,
        search_range=2,
        window_radius=4,
        realign_window=1,
        min_logfpr=None,
        cache="off",
        cache_dir=".mimosa-cache",
        pvalue=False,
        null_distribution=None,
        null_search_dirs=None,
        effective_number_of_targets=None,
    )

    kwargs = map_args_to_comparator_kwargs(args)
    config = create_comparator_config(**kwargs)

    assert kwargs["n_jobs"] == 3
    assert config["n_jobs"] == 3


@pytest.mark.parametrize(
    "key",
    ["n_permutations", "distortion_level", "min_kernel_size", "max_kernel_size", "permute_rows"],
)
def test_create_comparator_config_rejects_removed_null_options(key):
    """Removed Monte Carlo options should no longer be accepted by the public config."""
    with pytest.raises(ValueError, match="Unknown comparator option"):
        create_comparator_config(**{key: 1})


def test_create_comparator_config_validates_min_logfpr():
    """Profile floor should reject negative logFPR thresholds."""
    with pytest.raises(ValueError, match="min_logfpr"):
        create_comparator_config(min_logfpr=-0.1)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_create_comparator_config_validates_pfm_top_fraction(value):
    """PFM site-selection fraction should stay in the open-closed unit interval."""
    with pytest.raises(ValueError, match="pfm_top_fraction"):
        create_comparator_config(pfm_top_fraction=value)


def test_create_comparator_config_validates_cache_mode():
    """Profile cache mode should accept only explicit on/off values."""
    with pytest.raises(ValueError, match="cache_mode"):
        create_comparator_config(cache_mode="targets")

    config = create_comparator_config(cache_mode="on")
    assert config["cache_mode"] == "on"


def test_create_comparator_config_validates_profile_normalization():
    """Profile normalization mode should accept only known strategies."""
    with pytest.raises(ValueError, match="profile_normalization"):
        create_comparator_config(profile_normalization="zscore")

    config = create_comparator_config(profile_normalization="empirical_log_tail")
    assert config["profile_normalization"] == "empirical_log_tail"


def test_create_comparator_config_rejects_unknown_metric():
    """Comparator config should fail fast for unsupported metric names."""
    with pytest.raises(ValueError, match="metric must be one of"):
        create_comparator_config(metric="wrong")


def test_comparison_registry():
    """Test comparison registry functionality"""
    # Test that we can get registered strategies
    motif_strategy = comparison_registry.get("motif")
    assert motif_strategy is not None

    profile_strategy = comparison_registry.get("profile")
    assert profile_strategy is not None

    # Test invalid strategy returns None
    invalid_strategy = comparison_registry.get("invalid_strategy")
    assert invalid_strategy is None


def test_scan_model_with_pwm():
    """Test scanning with PWM model"""
    # Create simple PWM model
    representation = np.array(
        [
            [0.2, 0.3, 0.1],  # A
            [0.3, 0.2, 0.4],  # C
            [0.2, 0.4, 0.3],  # G
            [0.3, 0.1, 0.2],  # T
            [0.1, 0.1, 0.1],  # N (minimum values)
        ]
    )

    model = GenericModel(type_key="pwm", name="test_pwm", representation=representation, length=3, config={"kmer": 1})

    # Create test sequences
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),  # A,C,G,T,C,G,A
            np.array([1, 2, 3, 0], dtype=np.int8),  # C,G,T,A
        ]
    )

    # Test scanning
    scores = scan_model(model, sequences, "+")
    assert flatten_valid(scores).size > 0


def test_scan_model_strands_returns_strand_bundle():
    """Two-strand scanning should return one bundle with shape strands x rows x cols."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="test_pwm", representation=representation, length=3, config={"kmer": 1})
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),
            np.array([1, 2, 3, 0], dtype=np.int8),
        ]
    )

    strand_bundle = scan_model_strands(model, sequences)
    both_scores = scan_model(model, sequences, "both")
    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")

    assert strand_bundle["values"].shape[0] == 2
    np.testing.assert_array_equal(strand_bundle["lengths"], plus_scores["lengths"])
    np.testing.assert_allclose(strand_bundle["values"][PLUS_STRAND], plus_scores["values"])
    np.testing.assert_allclose(strand_bundle["values"][MINUS_STRAND], minus_scores["values"])
    np.testing.assert_allclose(both_scores["values"], strand_bundle["values"])
    np.testing.assert_array_equal(both_scores["lengths"], strand_bundle["lengths"])


def test_get_frequencies():
    """Test frequency calculation"""
    representation = np.array([[0.2, 0.3], [0.3, 0.2], [0.2, 0.4], [0.3, 0.1], [0.1, 0.1]])

    model = GenericModel("pwm", "test", representation, 2, {"kmer": 1})

    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3], dtype=np.int8),
            np.array([1, 2, 3, 0], dtype=np.int8),
        ]
    )

    frequencies = get_frequencies(model, sequences, "+")
    assert flatten_valid(frequencies).size > 0


def test_calculate_threshold_table_does_not_mutate_model_config():
    """Threshold-table calculation should stay pure and avoid storing runtime lookup tables on the model."""
    representation = np.array(
        [
            [1.0, 0.2],
            [0.2, 1.0],
            [0.1, 0.1],
            [0.1, 0.1],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="test_pwm", representation=representation, length=2, config={"kmer": 1})
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
            np.array([1, 0, 1, 0, 1, 0], dtype=np.int8),
        ]
    )

    table = calculate_threshold_table(model, sequences, strand="best")

    assert table.shape[1] == 2
    assert "_threshold_table" not in model.config
    assert "_threshold_tables" not in model.config


def test_calculate_threshold_table_both_uses_combined_strand_sample():
    """strand='both' should fit the threshold table on all + and - predictions."""
    representation = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(
        type_key="pwm",
        name="threshold_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1},
    )
    sequences = make_sequence_batch([_encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG")])

    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")
    expected = build_score_log_tail_table(
        np.concatenate((flatten_valid(plus_scores), flatten_valid(minus_scores)))
    ).astype(np.float64)

    table = calculate_threshold_table(model, sequences)

    np.testing.assert_allclose(table, expected)


def test_get_pfm_reconstructs_pwm_from_sites_with_single_pseudocount():
    """PFM reconstruction should ignore source PFM caches and add the pseudocount only once."""
    representation = np.array(
        [
            [5.0, 0.2],
            [0.1, 4.5],
            [0.1, 0.1],
            [0.1, 1.5],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    source_pfm = np.full((4, 2), 0.25, dtype=np.float32)
    model = GenericModel(
        type_key="pwm",
        name="test_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1, "_source_pfm": source_pfm},
    )
    sequences = make_sequence_batch(
        [
            _encode_sequence("AC"),
            _encode_sequence("AC"),
            _encode_sequence("AT"),
        ]
    )

    pfm = get_pfm(model, sequences, pseudocount=0.25)
    expected_pcm = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expected = pcm_to_pfm(expected_pcm, pseudocount=0.25).astype(np.float32)

    np.testing.assert_allclose(pfm, expected)
    assert not np.allclose(pfm, source_pfm)


def test_write_model_serializes_pwm_source_pfm(tmp_path):
    """PWM serialization should write the stored source PFM via the shared PFM writer."""
    source_pfm = np.array([[0.7, 0.1], [0.1, 0.7], [0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    pwm = pfm_to_pwm(source_pfm)
    representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
    model = GenericModel(
        type_key="pwm",
        name="serialized_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1, "_source_pfm": source_pfm},
    )

    path = tmp_path / "serialized.pfm"
    write_model(model, str(path))

    assert path.exists()
    written_pfm, length = read_pfm(str(path))
    assert length == 2
    np.testing.assert_allclose(written_pfm, source_pfm, atol=1e-6)


def test_read_model_rejects_pwm_pickle_without_source_pfm(tmp_path):
    """Legacy PWM pickles without a source PFM should no longer be accepted."""
    source_pfm = np.full((4, 2), 0.25, dtype=np.float32)
    pwm = pfm_to_pwm(source_pfm)
    representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
    legacy_model = GenericModel(
        type_key="pwm",
        name="legacy_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1, "_pfm": source_pfm},
    )
    path = tmp_path / "legacy_pwm.pkl"
    joblib.dump(legacy_model, path)

    with pytest.raises(ValueError, match="_source_pfm"):
        read_model(str(path), "pwm")


@pytest.mark.parametrize("model_type", ["sitega", "dimont", "slim"])
def test_read_model_rejects_non_model_pickle_payload(tmp_path, model_type):
    """Pickled payloads must always deserialize to GenericModel instances."""
    path = tmp_path / f"{model_type}.pkl"
    joblib.dump({"invalid": True}, path)

    with pytest.raises(TypeError, match="expected GenericModel"):
        read_model(str(path), model_type)


def test_write_dist_rejects_zero_score_range(tmp_path):
    """DIST output must reject degenerate score bounds to avoid inf values."""
    table = np.array([[0.5, 1.0], [0.4, 2.0]], dtype=np.float64)
    path = tmp_path / "invalid.dist"

    with pytest.raises(ValueError, match="max_score must be greater than min_score"):
        write_dist(table, max_score=1.0, min_score=1.0, path=str(path))


def test_get_sites_threshold_uses_current_sequences_by_default():
    """Thresholded site extraction should use the current sequences unless an external background is passed."""
    representation = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(
        type_key="pwm",
        name="threshold_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1},
    )
    sequences = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG")]
    )
    background = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG"), _encode_sequence("TG")]
    )

    best_sites = get_sites(model, sequences, mode="threshold", fpr_threshold=0.5, strand="best")
    both_sites = get_sites(model, sequences, mode="threshold", fpr_threshold=0.5)
    background_sites = get_sites(
        model,
        sequences,
        mode="threshold",
        fpr_threshold=0.5,
        strand="best",
        background_sequences=background,
    )

    assert best_sites["site"].tolist() == ["AC", "AC"]
    assert both_sites["site"].tolist() == ["AC", "AC", "AT", "AT"]
    assert both_sites["strand"].tolist() == ["+", "+", "+", "-"]
    assert background_sites["site"].tolist() == ["AC", "AC", "AT"]


def test_get_pfm_threshold_accepts_external_background_sequences():
    """PFM reconstruction should switch to external calibration only when requested."""
    representation = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(
        type_key="pwm",
        name="threshold_pwm",
        representation=representation,
        length=2,
        config={"kmer": 1},
    )
    sequences = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG")]
    )
    background = make_sequence_batch(
        [_encode_sequence("AC"), _encode_sequence("AT"), _encode_sequence("TG"), _encode_sequence("TG")]
    )

    default_pfm = get_pfm(model, sequences, mode="threshold", fpr_threshold=0.5, strand="best", pseudocount=0.25)
    background_pfm = get_pfm(
        model,
        sequences,
        mode="threshold",
        fpr_threshold=0.5,
        strand="best",
        background_sequences=background,
        pseudocount=0.25,
    )

    expected_default_pcm = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    expected_background_pcm = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    np.testing.assert_allclose(default_pfm, pcm_to_pfm(expected_default_pcm, pseudocount=0.25).astype(np.float32))
    np.testing.assert_allclose(background_pfm, pcm_to_pfm(expected_background_pcm, pseudocount=0.25).astype(np.float32))


def test_get_sites_best_skips_sequences_shorter_than_motif():
    """Best-site extraction should return no hits instead of crashing on short sequences."""
    representation = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="short_test", representation=representation, length=4, config={"kmer": 1})
    sequences = make_sequence_batch([_encode_sequence("ACG")])

    result = get_sites(model, sequences, mode="best")

    assert result.empty


def test_batch_all_scores_with_simple_data():
    """Test batch_all_scores with a simple dense masked batch."""
    data = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = make_sequence_batch([data[:3], data[3:]])
    representation = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = batch_all_scores(sequences, representation, kmer=1, is_revcomp=False)
    assert result["values"].shape == (2, 2)
    np.testing.assert_array_equal(result["lengths"], np.array([2, 2], dtype=np.int64))


def _manual_reverse_scores(seq: np.ndarray, matrix: np.ndarray, kmer: int, with_context: bool) -> np.ndarray:
    """Reproduce reverse-complement scoring with the legacy per-window logic."""
    motif_len = matrix.shape[-1]
    context_len = kmer - 1
    window_size = motif_len + context_len
    rc_table = np.array([3, 2, 1, 0, 4], dtype=np.int8)
    n_scores = max(0, seq.size - motif_len + 1)
    scores = np.zeros(n_scores, dtype=np.float32)

    for pos in range(n_scores):
        if with_context:
            window = np.full(window_size, 4, dtype=np.int8)
            for t in range(window_size):
                data_idx = pos + (window_size - 1 - t)
                if 0 <= data_idx < seq.size:
                    window[t] = rc_table[seq[data_idx]]
        else:
            window = rc_table[seq[pos : pos + motif_len][::-1]]
        scores[pos] = score_seq(window, kmer, matrix)

    return scores


def test_batch_all_scores_reverse_complement_preserves_positions():
    """Reverse-complement PWM scan should remain aligned to forward coordinates."""
    seq = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = make_sequence_batch([seq])
    representation = np.array(
        [
            [1.0, 0.1, 0.3],
            [0.2, 1.1, 0.2],
            [0.3, 0.2, 1.2],
            [0.4, 0.5, 0.6],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = batch_all_scores(sequences, representation, kmer=1, is_revcomp=True)
    expected = _manual_reverse_scores(seq, representation, kmer=1, with_context=False)

    np.testing.assert_allclose(row_values(result, 0), expected)


def test_batch_all_scores_reverse_complement_with_context_preserves_positions():
    """Reverse-complement BaMM scan should keep the same coordinate convention."""
    seq = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    sequences = make_sequence_batch([seq])
    representation = np.arange(25 * 3, dtype=np.float32).reshape(25, 3) / 10.0

    result = batch_all_scores(sequences, representation, kmer=2, is_revcomp=True, with_context=True)
    expected = _manual_reverse_scores(seq, representation, kmer=2, with_context=True)

    np.testing.assert_allclose(row_values(result, 0), expected)


@pytest.mark.parametrize(
    ("representation", "kmer", "with_context"),
    [
        (
            np.array(
                [
                    [1.0, 0.1, 0.3],
                    [0.2, 1.1, 0.2],
                    [0.3, 0.2, 1.2],
                    [0.4, 0.5, 0.6],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            1,
            False,
        ),
        (np.arange(25 * 3, dtype=np.float32).reshape(25, 3) / 10.0, 2, True),
    ],
)
def test_batch_all_scores_strands_matches_separate_calls(representation, kmer, with_context):
    """Two-strand batch scanning should match separate forward and reverse passes."""
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 0, 1], dtype=np.int8),
            np.array([3, 2, 1, 0, 3], dtype=np.int8),
        ]
    )

    expected_plus = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=False, with_context=with_context)
    expected_minus = batch_all_scores(sequences, representation, kmer=kmer, is_revcomp=True, with_context=with_context)
    plus_batch, minus_batch = batch_all_scores_strands(sequences, representation, kmer=kmer, with_context=with_context)

    np.testing.assert_allclose(plus_batch["values"], expected_plus["values"])
    np.testing.assert_array_equal(plus_batch["mask"], expected_plus["mask"])
    np.testing.assert_array_equal(plus_batch["lengths"], expected_plus["lengths"])
    np.testing.assert_allclose(minus_batch["values"], expected_minus["values"])
    np.testing.assert_array_equal(minus_batch["mask"], expected_minus["mask"])
    np.testing.assert_array_equal(minus_batch["lengths"], expected_minus["lengths"])


def test_rowwise_cosine_is_averaged_per_window():
    """Profile cosine should be the mean of per-window cosine values."""
    windows_1 = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 0.0], [1.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    np.testing.assert_allclose(
        rowwise_cosine(windows_1, windows_2),
        np.array([1.0, 0.0, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert score_window_collection("cosine", windows_1, windows_2) == pytest.approx(0.5)


def test_rowwise_co_is_averaged_per_window():
    """Profile rowwise CO should be the mean of per-window CO values."""
    windows_1 = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    np.testing.assert_allclose(
        rowwise_co(windows_1, windows_2),
        np.array([1.0, 0.5, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert score_window_collection("co_rowwise", windows_1, windows_2) == pytest.approx(0.75)


def test_rowwise_dice_is_averaged_per_window():
    """Profile rowwise Dice should be the mean of per-window Dice values."""
    windows_1 = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    np.testing.assert_allclose(
        rowwise_dice(windows_1, windows_2),
        np.array([1.0, 0.5, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert score_window_collection("dice_rowwise", windows_1, windows_2) == pytest.approx(0.75)


def test_overlap_profile_metrics_match_reference_formulas():
    """CO and Dice should use the weighted overlap formulas."""
    windows_1 = np.array([[1.0, 3.0], [2.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[2.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    intersection = np.minimum(windows_1, windows_2).sum()

    assert calc_co(windows_1, windows_2) == pytest.approx(intersection / min(windows_1.sum(), windows_2.sum()))
    assert calc_dice(windows_1, windows_2) == pytest.approx((2.0 * intersection) / (windows_1.sum() + windows_2.sum()))


@pytest.mark.parametrize(("metric", "expected"), [("co", 0.5), ("dice", 0.5)])
def test_window_metrics_score_only_selected_windows(metric, expected):
    """CO and Dice should be computed on the selected window collection only."""
    windows_1 = np.array([[2.0, 0.0]], dtype=np.float32)
    windows_2 = np.array([[1.0, 1.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    assert score_window_collection(metric, windows_1, windows_2) == pytest.approx(expected)


def test_dice_rowwise_differs_from_global_dice_when_window_weights_differ():
    """Rowwise Dice should weight windows uniformly, unlike global Dice."""
    windows_1 = np.array([[100.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    windows_2 = np.array([[100.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    score_window_collection = strategy_profile.__globals__["_score_window_collection"]

    assert calc_dice(windows_1, windows_2) == pytest.approx(202.0 / 204.0)
    assert score_window_collection("dice_rowwise", windows_1, windows_2) == pytest.approx(0.75)


def test_threshold_profile_selection_uses_or_logic():
    """Threshold mode should keep windows when either motif contributes the anchor."""
    compute_alignment = strategy_profile.__globals__["_compute_shifted_window_alignment"]
    collect_anchor_sites = strategy_profile.__globals__["_collect_anchor_sites"]
    lengths = np.array([3], dtype=np.int32)
    scores_1 = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    scores_2 = np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
    query_anchors = collect_anchor_sites(scores_1, lengths, 1.0)
    target_anchors = collect_anchor_sites(scores_2, lengths, 1.0)

    result = compute_alignment(
        scores_1,
        lengths,
        scores_2,
        lengths,
        0,
        np.array([0], dtype=np.int32),
        0,
        0,
        query_anchors,
        target_anchors,
        0,
        "co",
    )

    assert result["n_sites"] == 1
    assert result["score"] == pytest.approx(1.0)


def test_model2_threshold_anchors_are_realigned_on_model1():
    """Anchors from the second motif should be recentered on the best nearby site of the first motif."""
    collect_model2_candidates = strategy_profile.__globals__["_collect_model2_window_candidates"]
    collect_anchor_sites = strategy_profile.__globals__["_collect_anchor_sites"]
    lengths = np.array([5], dtype=np.int32)
    scores_1 = np.array([[0.0, 1.0, 3.0, 0.0, 0.0]], dtype=np.float32)
    scores_2 = np.array([[0.0, 4.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    anchor_rows, anchor_pos2 = collect_anchor_sites(scores_2, lengths, 1.0)

    rows, pos1, pos2 = collect_model2_candidates(
        scores_1,
        lengths,
        lengths,
        anchor_rows,
        anchor_pos2,
        0,
        0,
        0,
        1,
    )

    np.testing.assert_array_equal(rows, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(pos1, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(pos2, np.array([2], dtype=np.int32))


def test_profile_orientation_search_keeps_all_four_candidates():
    """Window-based profile search should still consider all four strand combinations."""
    score_profile_candidates = strategy_profile.__globals__["_score_profile_candidates"]
    query_plus = make_score_batch([np.array([1.0, 0.0, 0.0], dtype=np.float32)])
    query_minus = make_score_batch([np.array([0.0, 1.0, 0.0], dtype=np.float32)])
    target_plus = make_score_batch([np.array([0.0, 1.0, 0.0], dtype=np.float32)])
    target_minus = make_score_batch([np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    query_bundle = make_strand_bundle(query_plus, query_minus)
    target_bundle = make_strand_bundle(target_plus, target_minus)
    cfg = create_comparator_config(metric="co", search_range=0, window_radius=0, realign_window=0)

    best = strategy_profile.__globals__["_select_best_orientation"](
        score_profile_candidates(
            query_bundle,
            target_bundle,
            strategy_profile.__globals__["PROFILE_ORIENTATION_PAIRS"],
            cfg,
        )
    )

    assert best["orientation"] == "-+"
    assert best["score"] == pytest.approx(1.0)


def test_strategy_functions_exist():
    """Test that all strategy functions are properly defined"""
    # Test that strategy functions exist and are callable
    assert callable(strategy_motif)
    assert callable(strategy_profile)


def test_strategy_profile_uses_motif_offset_convention_for_score_tracks():
    """Profile strategy should report target position minus query position."""
    model1 = _make_scores_model("s1", [[0.0, 0.0, 9.0, 0.0, 0.0, 0.0]])
    model2 = _make_scores_model("s2", [[0.0, 0.0, 0.0, 0.0, 9.0, 0.0]])
    cfg = create_comparator_config(metric="co", search_range=4, window_radius=0, min_logfpr=0.1)

    forward = strategy_profile(model1, model2, None, cfg)
    reverse = strategy_profile(model2, model1, None, cfg)

    assert forward["orientation"] == "++"
    assert reverse["orientation"] == "++"
    assert forward["score"] == pytest.approx(1.0)
    assert reverse["score"] == pytest.approx(1.0)
    assert forward["offset"] == 2
    assert reverse["offset"] == -2
    assert forward["n_sites"] == 1
    assert reverse["n_sites"] == 1


def test_strategy_profile_offset_matches_motif_for_shifted_pwm_core():
    """Profile and motif strategies should use the same offset sign and value."""
    query = _make_shifted_core_pwm_model("query", core_offset=1)
    target = _make_shifted_core_pwm_model("target", core_offset=3)
    sequences = []
    embedded_site = _encode_sequence("ACG")
    for _ in range(64):
        sequence = np.full(60, _DNA_TO_INT["T"], dtype=np.int8)
        sequence[25 : 25 + embedded_site.size] = embedded_site
        sequences.append(sequence)
    sequence_batch = make_sequence_batch(sequences)
    motif_cfg = create_comparator_config(metric="cosine", pfm_mode=False)
    profile_cfg = create_comparator_config(
        metric="co",
        search_range=5,
        window_radius=0,
        realign_window=0,
    )

    motif_result = strategy_motif(query, target, sequence_batch, motif_cfg)
    profile_result = strategy_profile(query, target, sequence_batch, profile_cfg)
    motif_reverse = strategy_motif(target, query, sequence_batch, motif_cfg)
    profile_reverse = strategy_profile(target, query, sequence_batch, profile_cfg)

    assert motif_result["orientation"] == profile_result["orientation"] == "++"
    assert motif_result["offset"] == profile_result["offset"] == -2
    assert profile_result["score"] == pytest.approx(1.0)
    assert motif_reverse["orientation"] == profile_reverse["orientation"] == "++"
    assert motif_reverse["offset"] == profile_reverse["offset"] == 2
    assert profile_reverse["score"] == pytest.approx(1.0)


def test_strategy_profile_offset_matches_motif_for_reverse_complement_pwm_core():
    """Offset convention should also match for reverse-complement motif orientation."""
    core = (0, 1, 2, 3, 1, 0)
    target_core = tuple(int(base) for base in _reverse_complement_encoded(np.asarray(core, dtype=np.int8)))
    query = _make_shifted_core_pwm_model("query", core_offset=2, core=core, motif_length=12)
    target = _make_shifted_core_pwm_model("target", core_offset=3, core=target_core, motif_length=12)
    sequences = []
    embedded_site = np.asarray(core, dtype=np.int8)
    for _ in range(64):
        sequence = np.full(100, _DNA_TO_INT["A"], dtype=np.int8)
        sequence[45 : 45 + embedded_site.size] = embedded_site
        sequences.append(sequence)
    sequence_batch = make_sequence_batch(sequences)
    motif_cfg = create_comparator_config(metric="cosine", pfm_mode=False)
    profile_cfg = create_comparator_config(
        metric="co",
        search_range=8,
        window_radius=5,
        realign_window=0,
    )

    motif_result = strategy_motif(query, target, sequence_batch, motif_cfg)
    profile_result = strategy_profile(query, target, sequence_batch, profile_cfg)

    assert motif_result["orientation"] == profile_result["orientation"] == "+-"
    assert motif_result["offset"] == profile_result["offset"] == -1
    assert profile_result["score"] == pytest.approx(1.0)


def test_strategy_profile_empirical_uses_combined_strand_table():
    """Empirical profile normalization should use one combined +/- calibration table."""
    site = "CAGTAAACAG"
    rng = np.random.default_rng(0)
    encoded_site = _encode_sequence(site)
    sequences = []

    for _ in range(300):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    cfg = create_comparator_config(metric="co")

    plus_scores = scan_model(dimont, ragged_sequences, "+")
    minus_scores = scan_model(dimont, ragged_sequences, "-")
    combined_table = build_score_log_tail_table(
        np.concatenate((flatten_valid(plus_scores), flatten_valid(minus_scores)))
    )

    expected_plus = apply_score_log_tail_table(plus_scores, combined_table)
    expected_minus = apply_score_log_tail_table(minus_scores, combined_table)

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(dimont, ragged_sequences, ragged_sequences, cfg)

    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, PLUS_STRAND),
        flatten_valid(expected_plus),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, MINUS_STRAND),
        flatten_valid(expected_minus),
        atol=1e-6,
    )

    plus_slice = profile_row_values(resolved, PLUS_STRAND, 0)
    minus_slice = profile_row_values(resolved, MINUS_STRAND, 0)
    assert plus_slice[30] > minus_slice[29]


def test_strategy_profile_uses_background_for_empirical_calibration():
    """Profile normalization should use background scans when explicit calibration sequences are provided."""
    representation = np.array(
        [
            [2.0, -1.0],
            [-1.0, 2.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="m1", representation=representation, length=2, config={"kmer": 1})
    sequences = make_sequence_batch([_encode_sequence("ACACAC"), _encode_sequence("CAAAAA")])
    background = make_sequence_batch([_encode_sequence("AAAAAA"), _encode_sequence("CCCCCC")])
    cfg = create_comparator_config(metric="co", background=background)

    plus_scores = scan_model(model, sequences, "+")
    background_plus = scan_model(model, background, "+")
    background_minus = scan_model(model, background, "-")
    table = build_score_log_tail_table(
        np.concatenate((flatten_valid(background_plus), flatten_valid(background_minus)))
    )
    expected_plus = apply_score_log_tail_table(plus_scores, table)

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(model, sequences, background, cfg)

    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, PLUS_STRAND),
        flatten_valid(expected_plus),
        atol=1e-6,
    )


def test_resolve_profile_bundle_matches_direct_two_strand_normalization():
    """Resolved profile bundle should match direct two-strand normalization exactly."""
    representation = np.array(
        [
            [1.2, -0.3],
            [-0.4, 1.0],
            [-0.5, -0.4],
            [-0.3, -0.5],
            [-0.5, -0.5],
        ],
        dtype=np.float32,
    )
    model = GenericModel(type_key="pwm", name="joint", representation=representation, length=2, config={"kmer": 1})
    sequences = make_sequence_batch(
        [
            _encode_sequence("ACGTAC"),
            _encode_sequence("TGCATG"),
            _encode_sequence("AAAAAC"),
        ]
    )
    cfg = create_comparator_config(metric="co", cache_mode="off")

    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    resolved = resolve_profile_bundle(model, sequences, sequences, cfg)

    plus_scores = scan_model(model, sequences, "+")
    minus_scores = scan_model(model, sequences, "-")
    expected_plus, expected_minus = normalize_empirical_log_tail_pair(plus_scores, minus_scores)

    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, PLUS_STRAND),
        flatten_valid(expected_plus),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flatten_profile_bundle(resolved, MINUS_STRAND),
        flatten_valid(expected_minus),
        atol=1e-6,
    )


def test_strategy_motif_handles_reverse_complement_for_higher_order_tensors():
    """Higher-order tensor comparison should reverse both nucleotide values and context-axis order."""
    rng = np.random.default_rng(7)
    core = rng.normal(size=(4, 4, 6)).astype(np.float32)
    rep1 = np.full((5, 5, 6), -3.0, dtype=np.float32)
    rep1[:4, :4, :] = core

    core_rc = np.transpose(core, (1, 0, 2))[::-1, ::-1, ::-1]
    rep2 = np.full((5, 5, 6), -3.0, dtype=np.float32)
    rep2[:4, :4, :] = core_rc

    model1 = GenericModel(type_key="sitega", name="m1", representation=rep1, length=6, config={"kmer": 2})
    model2 = GenericModel(type_key="sitega", name="m2", representation=rep2, length=6, config={"kmer": 2})

    result = strategy_motif(model1, model2, None, create_comparator_config(metric="cosine"))

    assert result["orientation"] == "+-"
    assert result["score"] == pytest.approx(1.0, abs=1e-6)


def test_foxa1_cross_type_motif_comparison_recovers_dimont_and_slim_similarity():
    """FOXA1 higher-order models should stay aligned with PWM/BaMM after PFM reconstruction."""
    rng = np.random.default_rng(1)
    sequences = make_sequence_batch([rng.integers(0, 4, size=40, dtype=np.int8) for _ in range(2000)])

    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    bamm = read_model(str(FIXTURES_ROOT / "bamm" / "PEAKS036274_FOXA1_P35582_MACS2_motif_1.ihbcp"), "bamm", order=2)
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    slim = read_model(str(FIXTURES_ROOT / "slim" / "PEAKS036274_FOXA1_P35582_MACS2-model-2.xml"), "slim")
    cfg = create_comparator_config(metric="pcc")

    pwm_dimont = strategy_motif(pwm, dimont, sequences, cfg)
    pwm_slim = strategy_motif(pwm, slim, sequences, cfg)
    bamm_dimont = strategy_motif(bamm, dimont, sequences, cfg)
    bamm_slim = strategy_motif(bamm, slim, sequences, cfg)
    bamm_pwm = strategy_motif(bamm, pwm, sequences, cfg)

    assert pwm_dimont["orientation"] == "+-"
    assert pwm_dimont["score"] > 0.60

    assert pwm_slim["orientation"] == "++"
    assert pwm_slim["score"] > 0.68

    assert bamm_dimont["orientation"] == "+-"
    assert bamm_dimont["score"] > 0.54

    assert bamm_slim["orientation"] == "++"
    assert bamm_slim["score"] > 0.65

    assert bamm_pwm["orientation"] == "--"
    assert bamm_pwm["score"] > 0.85


def test_strategy_profile_co_uses_sparse_signal_on_foxa1_dimont_pwm_sites():
    """CO profile comparison should stay stable for FOXA1 profiles after strand normalization."""
    site = "CCAGAGTAAACAG"
    dna_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    rng = np.random.default_rng(0)
    sequences = []
    encoded_site = np.array([dna_to_int[base] for base in site], dtype=np.int8)

    for _ in range(500):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")

    result = strategy_profile(dimont, pwm, ragged_sequences, create_comparator_config(metric="co"))

    assert result["orientation"] == "--"
    assert abs(result["offset"]) == 2
    assert result["score"] > 0.55


def test_profile_orientation_search_requires_minus_minus_candidate_on_real_profiles():
    """Real profile bundles can prefer -- over the reduced ++ / +- search space."""
    site = "CCAGAGTAAACAG"
    rng = np.random.default_rng(0)
    sequences = []
    encoded_site = _encode_sequence(site)

    for _ in range(500):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    cfg = create_comparator_config(metric="co")
    calibration_sequences = strategy_profile.__globals__["_get_profile_background_sequences"](ragged_sequences, cfg)
    resolve_profile_bundle = strategy_profile.__globals__["_resolve_profile_bundle"]
    score_profile_candidates = strategy_profile.__globals__["_score_profile_candidates"]
    orientation_pairs = strategy_profile.__globals__["PROFILE_ORIENTATION_PAIRS"]

    pwm_bundle = resolve_profile_bundle(pwm, ragged_sequences, calibration_sequences, cfg)
    dimont_bundle = resolve_profile_bundle(dimont, ragged_sequences, calibration_sequences, cfg)
    candidates = score_profile_candidates(pwm_bundle, dimont_bundle, orientation_pairs, cfg)
    score_by_orientation = {candidate["orientation"]: float(candidate["score"]) for candidate in candidates}

    assert max(score_by_orientation, key=score_by_orientation.get) == "--"
    assert score_by_orientation["--"] > max(score_by_orientation["++"], score_by_orientation["+-"])


def test_strategy_profile_is_symmetric_when_models_peak_on_different_strands():
    """Profile comparison should stay numerically close when query and target are swapped."""
    site = "CAGTAAACAG"
    dna_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    rng = np.random.default_rng(0)
    sequences = []
    encoded_site = np.array([dna_to_int[base] for base in site], dtype=np.int8)

    for _ in range(300):
        seq = rng.integers(0, 4, size=80, dtype=np.int8)
        seq[30 : 30 + encoded_site.size] = encoded_site
        sequences.append(seq)

    ragged_sequences = make_sequence_batch(sequences)
    pwm = read_model(str(FIXTURES_ROOT / "pwm" / "PEAKS036274_FOXA1_P35582_MACS2.meme"), "pwm")
    dimont = read_model(str(FIXTURES_ROOT / "dimont" / "PEAKS036274_FOXA1_P35582_MACS2-model-1.xml"), "dimont")
    cfg = create_comparator_config(metric="co")

    pwm_vs_dimont = strategy_profile(pwm, dimont, ragged_sequences, cfg)
    dimont_vs_pwm = strategy_profile(dimont, pwm, ragged_sequences, cfg)

    assert pwm_vs_dimont["score"] == pytest.approx(dimont_vs_pwm["score"], abs=1e-3)
    assert pwm_vs_dimont["orientation"] == dimont_vs_pwm["orientation"] == "--"
    assert pwm_vs_dimont["offset"] == -dimont_vs_pwm["offset"]
    assert pwm_vs_dimont["score"] > 0.6


def test_strategy_runtime_cache_keys_use_model_content_not_name():
    """Runtime cache should not mix distinct models sharing the same name."""

    def build_model(seed: int, name: str) -> GenericModel:
        rng = np.random.default_rng(seed)
        pfm = rng.random((4, 6), dtype=np.float32)
        pfm /= pfm.sum(axis=0, keepdims=True)
        pwm = pfm_to_pwm(pfm)
        representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
        return GenericModel("pwm", name, representation, 6, {"kmer": 1, "_source_pfm": pfm})

    sequences = make_sequence_batch(
        [np.random.default_rng(3).integers(0, 4, size=200, dtype=np.int8) for _ in range(20)]
    )
    cfg_profile = create_comparator_config(metric="co")
    cfg_motif = create_comparator_config(metric="cosine", pfm_mode=False)

    model_a = build_model(1, "a")
    model_b = build_model(2, "b")
    model_same_1 = build_model(1, "same")
    model_same_2 = build_model(2, "same")

    profile_named = strategy_profile(model_a, model_b, sequences, cfg_profile)
    profile_same_name = strategy_profile(model_same_1, model_same_2, sequences, cfg_profile)
    motif_named = strategy_motif(model_a, model_b, sequences, cfg_motif)
    motif_same_name = strategy_motif(model_same_1, model_same_2, sequences, cfg_motif)

    assert profile_same_name["score"] == pytest.approx(profile_named["score"])
    assert motif_same_name["score"] == pytest.approx(motif_named["score"])


def test_strategy_profile_uses_disk_cache_for_target_and_query(tmp_path, monkeypatch):
    """Cached query and target profiles should be reused across repeated comparisons."""

    def make_model(name: str) -> GenericModel:
        representation = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.2, 0.8, 0.3],
                [0.1, 0.3, 0.9],
                [0.3, 0.2, 0.1],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        return GenericModel(type_key="pwm", name=name, representation=representation, length=3, config={"kmer": 1})

    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int8),
            np.array([1, 2, 3, 0, 1, 2, 3], dtype=np.int8),
        ]
    )
    query = make_model("query")
    target = make_model("target")
    cfg = create_comparator_config(metric="co", cache_mode="on", cache_dir=str(tmp_path))

    first = strategy_profile(query, target, sequences, cfg)
    assert first["target"] == "target"
    assert len(list(tmp_path.rglob("*.npz"))) == 2

    fresh_query = make_model("query")
    fresh_target = make_model("target")
    original_scan_strands = strategy_profile.__globals__["scan_model_strands"]

    def guarded_scan_strands(model, current_sequences):
        if model.name == "query":
            raise AssertionError("query scan should be served from disk cache")
        if model.name == "target":
            raise AssertionError("target scan should be served from disk cache")
        return original_scan_strands(model, current_sequences)

    monkeypatch.setitem(strategy_profile.__globals__, "scan_model_strands", guarded_scan_strands)
    second = strategy_profile(fresh_query, fresh_target, sequences, cfg)

    assert second["target"] == "target"
    assert second["score"] == pytest.approx(first["score"])
    assert second["orientation"] == first["orientation"]


def test_strategy_profile_uses_batched_orientation_scoring(monkeypatch):
    """Profile strategy should score the four observed orientation pairs."""
    scores_1 = _score_batch_from_flat(
        np.array([0.0, 0.8, 1.7, 0.2, 0.0, 1.1, 0.4, 0.0], dtype=np.float32),
        np.array([0, 4, 8], dtype=np.int64),
    )
    scores_2 = _score_batch_from_flat(
        np.array([0.0, 1.0, 1.6, 0.3, 0.0, 0.9, 0.5, 0.0], dtype=np.float32),
        np.array([0, 4, 8], dtype=np.int64),
    )
    model1 = GenericModel(
        type_key="scores",
        name="q",
        representation=None,
        length=0,
        config={"scores_data": scores_1},
    )
    model2 = GenericModel(
        type_key="scores",
        name="t",
        representation=None,
        length=0,
        config={"scores_data": scores_2},
    )
    cfg = create_comparator_config(metric="co", n_jobs=1, search_range=2)

    call_sizes = []
    original_candidates = strategy_profile.__globals__["_score_profile_candidates"]

    def recording_candidates(left_bundle, right_bundle, pair_specs, cfg):
        call_sizes.append(len(pair_specs))
        return original_candidates(left_bundle, right_bundle, pair_specs, cfg)

    monkeypatch.setitem(
        strategy_profile.__globals__,
        "_score_profile_candidates",
        recording_candidates,
    )

    result = strategy_profile(model1, model2, None, cfg)

    assert result["score"] >= 0.0
    assert 4 in call_sizes


def test_clear_cache_removes_cached_profiles(tmp_path):
    """Cache cleanup helper should remove stored profile artifacts."""
    representation = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.2, 0.8, 0.3],
            [0.1, 0.3, 0.9],
            [0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sequences = make_sequence_batch([np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int8)])
    query = GenericModel(type_key="pwm", name="query", representation=representation, length=3, config={"kmer": 1})
    target = GenericModel(type_key="pwm", name="target", representation=representation, length=3, config={"kmer": 1})
    cfg = create_comparator_config(metric="co", cache_mode="on", cache_dir=str(tmp_path))

    strategy_profile(query, target, sequences, cfg)

    removed = clear_cache(str(tmp_path))

    assert removed > 0
    assert not tmp_path.exists()


def test_create_one_to_one_config_builds_unified_config():
    """Unified config builder should create comparator config from kwargs."""
    config = create_one_to_one_config(
        query="a.meme",
        target="b.pfm",
        query_type="pwm",
        target_type="pwm",
        strategy="profile",
        metric="co",
        n_jobs=2,
        seed=99,
    )

    assert config["strategy"] == "profile"
    assert config["comparator"]["metric"] == "co"
    assert config["comparator"]["n_jobs"] == 2
    assert config["seed"] == 99


def test_create_one_to_many_config_builds_unified_config():
    """Unified one-vs-many config builder should create comparator config from kwargs."""
    config = create_one_to_many_config(
        query="query.meme",
        targets=["target_a.pfm", "target_b.pfm"],
        query_type="pwm",
        target_type="pwm",
        strategy="profile",
        metric="co",
        n_jobs=3,
        seed=11,
    )

    assert config["strategy"] == "profile"
    assert config["comparator"]["metric"] == "co"
    assert config["comparator"]["n_jobs"] == 3
    assert config["targets"] == ("target_a.pfm", "target_b.pfm")
    assert config["seed"] == 11


def test_run_one_to_one_with_unified_config_and_models():
    """run_one_to_one should work with preloaded GenericModel objects."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model1 = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    model2 = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})
    sequences = make_sequence_batch(
        [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8),
            np.array([1, 2, 3, 0, 1, 2], dtype=np.int8),
        ]
    )

    config = create_one_to_one_config(
        query=model1,
        target=model2,
        strategy="profile",
        sequences=sequences,
        metric="co",
        seed=7,
    )
    result = run_one_to_one(config)

    assert "score" in result
    assert "offset" in result
    assert "orientation" in result


def test_run_one_to_many_matches_pairwise_profile_results():
    """One-vs-many profile API should match repeated pairwise comparisons."""
    query_scores = _score_batch_from_flat(np.array([0.2, 0.5, 0.8], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    target_a_scores = _score_batch_from_flat(
        np.array([0.2, 0.4, 0.9], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_b_scores = _score_batch_from_flat(
        np.array([0.8, 0.5, 0.2], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    query = GenericModel(
        type_key="scores",
        name="query",
        representation=None,
        length=0,
        config={"scores_data": query_scores},
    )
    target_a = GenericModel(
        type_key="scores",
        name="target_a",
        representation=None,
        length=0,
        config={"scores_data": target_a_scores},
    )
    target_b = GenericModel(
        type_key="scores",
        name="target_b",
        representation=None,
        length=0,
        config={"scores_data": target_b_scores},
    )

    config = create_one_to_many_config(
        query=query,
        targets=[target_a, target_b],
        strategy="profile",
        metric="co",
    )
    results = run_one_to_many(config)

    expected_a = run_one_to_one(create_one_to_one_config(query=query, target=target_a, strategy="profile", metric="co"))
    expected_b = run_one_to_one(create_one_to_one_config(query=query, target=target_b, strategy="profile", metric="co"))

    assert [result["target"] for result in results] == ["target_a", "target_b"]
    for result, expected in zip(results, [expected_a, expected_b], strict=False):
        assert result["query"] == expected["query"]
        assert result["target"] == expected["target"]
        assert result["orientation"] == expected["orientation"]
        assert result["offset"] == expected["offset"]
        np.testing.assert_allclose(result["score"], expected["score"])


def test_run_one_to_many_passes_targets_lazily(monkeypatch):
    """One-vs-many executor should receive a lazy target iterable instead of a materialized list."""
    query_scores = _score_batch_from_flat(
        np.array([0.2, 0.5, 0.8], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_scores = _score_batch_from_flat(
        np.array([0.2, 0.4, 0.9], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    query = GenericModel(
        type_key="scores",
        name="query",
        representation=None,
        length=0,
        config={"scores_data": query_scores},
    )
    target = GenericModel(
        type_key="scores",
        name="target",
        representation=None,
        length=0,
        config={"scores_data": target_scores},
    )
    config = create_one_to_many_config(query=query, targets=[target], strategy="profile", metric="co")
    observed = {}

    def fake_compare_one_to_many_models(query_model, target_models, strategy, config, sequences=None, background=None):
        observed["is_list"] = isinstance(target_models, list)
        materialized = list(target_models)
        observed["count"] = len(materialized)
        return [
            ComparisonResult(
                query=query_model.name,
                target=materialized[0].name,
                score=1.0,
                offset=0,
                orientation="++",
                metric="co",
            )
        ]

    monkeypatch.setattr(api_module, "compare_one_to_many_models", fake_compare_one_to_many_models)

    results = run_one_to_many(config)

    assert observed == {"is_list": False, "count": 1}
    assert results == [
        ComparisonResult(query="query", target="target", score=1.0, offset=0, orientation="++", metric="co")
    ]


def test_run_one_to_many_preserves_generator_targets():
    """One-vs-many API should not lose targets when config["targets"] is a generator."""
    query_scores = _score_batch_from_flat(
        np.array([0.2, 0.5, 0.8], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_a_scores = _score_batch_from_flat(
        np.array([0.1, 0.3, 0.9], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    target_b_scores = _score_batch_from_flat(
        np.array([0.6, 0.2, 0.4], dtype=np.float32),
        np.array([0, 3], dtype=np.int64),
    )
    query = GenericModel("scores", "query", None, 0, {"scores_data": query_scores})
    target_a = GenericModel("scores", "target_a", None, 0, {"scores_data": target_a_scores})
    target_b = GenericModel("scores", "target_b", None, 0, {"scores_data": target_b_scores})

    config = create_one_to_many_config(
        query=query,
        targets=[target_a, target_b],
        strategy="profile",
        metric="co",
    )
    results = run_one_to_many(config)

    assert [item["target"] for item in results] == ["target_a", "target_b"]


@pytest.mark.parametrize("metric", ["corr", "cj", "l1sim"])
def test_run_one_to_one_rejects_removed_profile_metrics(metric):
    """Profile mode should reject unsupported legacy metrics."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model1 = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    model2 = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})
    sequences = make_sequence_batch([np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8)])

    with pytest.raises(ValueError, match="metric"):
        config = create_one_to_one_config(
            query=model1,
            target=model2,
            strategy="profile",
            sequences=sequences,
            metric=metric,
            seed=7,
        )
        run_one_to_one(config)


def test_strategy_profile_accepts_background_configuration():
    """Profile strategy should accept external background calibration sequences."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.2, 0.3, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    background = make_sequence_batch([np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)])
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric="co", background=background)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["metric"] == "co"
    assert "score" in result


def test_strategy_profile_co_has_no_default_floor():
    """CO should not apply an implicit logFPR floor when min_logfpr is omitted."""
    scores_1 = _score_batch_from_flat(np.array([0.2, 0.2, 0.1], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.2, 0.2, 0.1], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    result = strategy_profile(
        model1,
        model2,
        None,
        create_comparator_config(metric="co", window_radius=0),
    )

    assert result["score"] == pytest.approx(1.0)


def test_strategy_profile_zero_min_logfpr_uses_best_anchor_mode():
    """min_logfpr=0 should behave like an omitted threshold."""
    scores_1 = _score_batch_from_flat(
        np.array([0.1, 0.9, 0.8, 0.7], dtype=np.float32),
        np.array([0, 4], dtype=np.int64),
    )
    scores_2 = _score_batch_from_flat(
        np.array([0.2, 0.95, 0.1, 0.1], dtype=np.float32),
        np.array([0, 4], dtype=np.int64),
    )
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    kwargs = {"metric": "co", "window_radius": 0, "search_range": 0}
    omitted = strategy_profile(model1, model2, None, create_comparator_config(**kwargs, min_logfpr=None))
    zero = strategy_profile(model1, model2, None, create_comparator_config(**kwargs, min_logfpr=0))

    assert zero["score"] == pytest.approx(omitted["score"])
    assert zero["n_sites"] == omitted["n_sites"] == 1


@pytest.mark.parametrize("metric", ["co", "co_rowwise", "dice", "dice_rowwise", "cosine"])
def test_strategy_profile_handles_all_positions_masked_by_threshold(metric):
    """Threshold site selection should not crash when no sites survive the cutoff."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.1, 0.2, 0.4], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})
    cfg = create_comparator_config(metric=metric, min_logfpr=10.0)

    result = strategy_profile(model1, model2, None, cfg)

    assert result["score"] == pytest.approx(0.0)
    assert result["offset"] == 0
    assert result["orientation"] == "++"
    assert result["n_sites"] == 0


def test_run_one_to_one_supports_dice_for_profile():
    """Unified API should expose the Dice profile metric."""
    scores_1 = _score_batch_from_flat(np.array([0.1, 0.5, 1.0], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    scores_2 = _score_batch_from_flat(np.array([0.1, 0.5, 0.9], dtype=np.float32), np.array([0, 3], dtype=np.int64))
    model1 = GenericModel(type_key="scores", name="s1", representation=None, length=0, config={"scores_data": scores_1})
    model2 = GenericModel(type_key="scores", name="s2", representation=None, length=0, config={"scores_data": scores_2})

    config = create_one_to_one_config(query=model1, target=model2, strategy="profile", metric="dice", seed=7)
    result = run_one_to_one(config)

    assert result["metric"] == "dice"
    assert 0.0 <= result["score"] <= 1.0


def test_run_one_to_one_supports_dice_rowwise_for_profile():
    """Unified API should expose the window-averaged rowwise Dice profile metric."""
    model1 = _make_scores_model("s1", [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    model2 = _make_scores_model("s2", [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])

    config = create_one_to_one_config(
        query=model1,
        target=model2,
        strategy="profile",
        metric="dice_rowwise",
        seed=7,
        window_radius=1,
    )
    result = run_one_to_one(config)

    assert result["metric"] == "dice_rowwise"
    assert 0.0 <= result["score"] <= 1.0


def test_run_one_to_one_supports_cosine_for_profile():
    """Unified API should expose the window-averaged cosine profile metric."""
    model1 = _make_scores_model("s1", [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    model2 = _make_scores_model("s2", [[0.0, 1.0, 0.0], [0.0, 1.0, -1.0]])

    config = create_one_to_one_config(
        query=model1,
        target=model2,
        strategy="profile",
        metric="cosine",
        seed=7,
        window_radius=1,
    )
    result = run_one_to_one(config)

    assert result["metric"] == "cosine"
    assert 0.0 <= result["score"] <= 1.0


def test_run_one_to_one_supports_co_rowwise_for_profile():
    """Unified API should expose the window-averaged rowwise CO profile metric."""
    model1 = _make_scores_model("s1", [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    model2 = _make_scores_model("s2", [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])

    config = create_one_to_one_config(
        query=model1,
        target=model2,
        strategy="profile",
        metric="co_rowwise",
        seed=7,
        window_radius=1,
    )
    result = run_one_to_one(config)

    assert result["metric"] == "co_rowwise"
    assert 0.0 <= result["score"] <= 1.0


def test_compare_motifs_shortcut_works_with_single_import_api():
    """compare_motifs should provide one-call high-level API."""
    representation = np.array(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3],
            [0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    model1 = GenericModel(type_key="pwm", name="m1", representation=representation, length=3, config={"kmer": 1})
    model2 = GenericModel(type_key="pwm", name="m2", representation=representation, length=3, config={"kmer": 1})
    sequences = make_sequence_batch([np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int8)])

    result = compare_motifs(
        model1=model1,
        model2=model2,
        strategy="profile",
        sequences=sequences,
        metric="co",
        seed=13,
    )

    assert result["query"] == "m1"
    assert result["target"] == "m2"


def test_compare_one_to_many_matches_pairwise_motif_results():
    """One-vs-many motif API should match repeated pairwise comparisons."""
    query_representation = np.array(
        [
            [0.6, 0.1, 0.3],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.6],
            [0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_a_representation = np.array(
        [
            [0.55, 0.15, 0.3],
            [0.25, 0.65, 0.1],
            [0.1, 0.1, 0.55],
            [0.1, 0.1, 0.05],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_b_representation = np.array(
        [
            [0.1, 0.6, 0.1],
            [0.6, 0.1, 0.1],
            [0.1, 0.1, 0.6],
            [0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = GenericModel(
        type_key="pwm",
        name="query",
        representation=query_representation,
        length=3,
        config={"kmer": 1},
    )
    target_a = GenericModel(
        type_key="pwm",
        name="target_a",
        representation=target_a_representation,
        length=3,
        config={"kmer": 1},
    )
    target_b = GenericModel(
        type_key="pwm",
        name="target_b",
        representation=target_b_representation,
        length=3,
        config={"kmer": 1},
    )

    results = compare_one_to_many(
        query=query,
        targets=[target_a, target_b],
        strategy="motif",
        metric="pcc",
    )

    expected_a = compare_motifs(query, target_a, strategy="motif", metric="pcc")
    expected_b = compare_motifs(query, target_b, strategy="motif", metric="pcc")

    assert [result["target"] for result in results] == ["target_a", "target_b"]
    for result, expected in zip(results, [expected_a, expected_b], strict=False):
        assert result["orientation"] == expected["orientation"]
        assert result["offset"] == expected["offset"]
        np.testing.assert_allclose(result["score"], expected["score"])
