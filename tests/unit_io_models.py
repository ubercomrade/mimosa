# ruff: noqa: F403,F405

from tests.unit_support import *


def test_read_scores_basic(tmp_path):
    """Numerical score profiles should be parsed from FASTA-like input."""
    path = tmp_path / "scores.fasta"
    path.write_text(">seq1\n0.1 0.2 0.3\n>seq2\n1.0 2.0\n", encoding="utf-8")

    result = read_scores(path)

    assert len(result["lengths"]) == 2
    np.testing.assert_allclose(row_values(result, 0), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(row_values(result, 1), np.array([1.0, 2.0], dtype=np.float32))


def test_read_slim_rejects_missing_required_arrays(tmp_path):
    """Malformed SLIM XML should fail with a domain error instead of int(None)/TypeError."""
    path = tmp_path / "broken-slim.xml"
    path.write_text(
        "<root><SLIM><length>1</length><distance>0</distance></SLIM></root>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SLIM componentMixtureParameters"):
        read_slim(str(path))


def test_read_meme_rejects_motif_without_name(tmp_path):
    """MEME parser should reject nameless MOTIF records with a domain error."""
    path = tmp_path / "nameless.meme"
    path.write_text("MOTIF\nletter-probability matrix: alength= 4 w= 1\n0.25 0.25 0.25 0.25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MOTIF line has no name"):
        read_meme(str(path))


def test_read_meme_many_rejects_short_matrix(tmp_path):
    """MEME parser should reject matrices shorter than the declared motif length."""
    path = tmp_path / "short.meme"
    path.write_text(
        "MOTIF bad\nletter-probability matrix: alength= 4 w= 2\n0.25 0.25 0.25 0.25\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 2 rows with 4 columns"):
        read_meme_many(path)


def test_read_pfm_rejects_invalid_shape(tmp_path):
    """PFM parser should require a nucleotide axis."""
    path = tmp_path / "invalid.pfm"
    path.write_text("0.1 0.2 0.3\n0.3 0.2 0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="one axis must contain 4 or 5"):
        read_pfm(str(path))


def test_read_sitega_rejects_out_of_range_segment(tmp_path):
    """SiteGA parser should validate segment bounds against model length."""
    path = tmp_path / "invalid.mat"
    path.write_text("toy\n1\tLPD count\n2\tModel length\n0\tMinimum\n1\tRazmah\n0\t2\t1.0\t0\tac\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside model length"):
        read_sitega(str(path))


def test_parse_file_content_rejects_invalid_bamm_width(tmp_path):
    """BaMM parser should validate expected order widths."""
    path = tmp_path / "invalid.ihbcp"
    path.write_text("0.25 0.25 0.25\n\n0.25 0.25 0.25 0.25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 4"):
        parse_file_content(str(path))


def test_format_params_basic():
    """Test basic parameter formatting"""
    params = {"k": 3, "metric": "pcc", "n_perm": 1000}
    formatted = format_params(params)

    # Should be in alphabetical order: k-3_metric-pcc_n_perm-1000
    expected = "k-3_metric-pcc_n_perm-1000"
    assert formatted == expected


def test_generic_model_creation():
    """Test GenericModel creation and basic field wiring."""
    representation = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.1, 0.2]])

    model = GenericModel(type_key="pwm", name="test_model", representation=representation, length=2, config={"kmer": 1})

    assert model.type_key == "pwm"
    assert model.name == "test_model"
    assert model.length == 2
    assert model.config["kmer"] == 1


def test_model_registry():
    """Test model registry functionality"""
    # Test that we can get registered strategies
    pwm_strategy = model_registry.get("pwm")
    assert pwm_strategy is not None
    assert isinstance(pwm_strategy, ModelHandler)
    assert callable(pwm_strategy.scan)
    assert callable(pwm_strategy.load)
    assert callable(pwm_strategy.write)
    assert callable(pwm_strategy.score_bounds)

    bamm_strategy = model_registry.get("bamm")
    assert bamm_strategy is not None

    dimont_strategy = model_registry.get("dimont")
    assert dimont_strategy is not None

    slim_strategy = model_registry.get("slim")
    assert slim_strategy is not None


def test_top_level_exports_validate_metric():
    """Top-level package should expose the public metric validator."""
    assert mimosa.validate_metric("PCC") == "pcc"
    assert "validate_metric" in mimosa.__all__


def test_read_model_supports_dimont_xml_and_matches_example_score():
    """Dimont XML models should load and reproduce Java raw-score log-odds semantics."""
    path = FIXTURES_ROOT / "dimont" / "exampleD-model-1.xml"
    plus_site = _encode_sequence("TTCCAGGGAACCC")
    model = read_model(str(path), "dimont")
    plus_sequence = make_sequence_batch([plus_site])
    minus_sequence = make_sequence_batch([_encode_sequence("GGGTTCCCTGGAA")])

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "dimont"
    assert model.length == 13
    assert model.config["kmer"] == 1
    assert model.representation.shape == (5, 13)
    assert row_values(plus_scores, 0)[0] == pytest.approx(_reference_dimont_site_score(path, plus_site))
    assert row_values(minus_scores, 0)[0] == pytest.approx(_reference_dimont_site_score(path, plus_site))


def test_read_model_supports_higher_order_dimont_xml():
    """Higher-order Dimont XML models should scan via the shared context kernel in log-odds space."""
    path = FIXTURES_ROOT / "dimont" / "stat_dimont-model-1.xml"
    model = read_model(str(path), "dimont")
    plus_sequence = make_sequence_batch([_encode_sequence("AACCC")])
    minus_sequence = make_sequence_batch([_encode_sequence("GGGTT")])

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "dimont"
    assert model.length == 5
    assert model.config["kmer"] == 4
    assert model.representation.shape == (5, 5, 5, 5, 5)
    assert row_values(plus_scores, 0)[0] == pytest.approx(_reference_dimont_site_score(path, _encode_sequence("AACCC")))
    assert row_values(minus_scores, 0)[0] == pytest.approx(
        _reference_dimont_site_score(path, _encode_sequence("AACCC"))
    )


def test_read_model_supports_slim_xml_and_matches_example_score():
    """Slim XML models should reproduce the exact uniform-background log-odds site scores."""
    model = read_model(str(FIXTURES_ROOT / "slim" / "example-model-1.xml"), "slim")
    plus_sequence = make_sequence_batch([_encode_sequence("TTCCTCGGAACTGAG")])
    minus_sequence = make_sequence_batch([_encode_sequence("CTCAGTTCCGAGGAA")])

    plus_scores = scan_model(model, plus_sequence, "+")
    minus_scores = scan_model(model, minus_sequence, "-")

    assert model.type_key == "slim"
    assert model.length == 15
    assert model.config["kmer"] == 6
    assert model.representation.shape == (5, 5, 5, 5, 5, 5, 15)
    assert row_values(plus_scores, 0)[0] == pytest.approx(6.243921739188615, abs=2e-5)
    assert row_values(minus_scores, 0)[0] == pytest.approx(6.243921739188615, abs=2e-5)


@pytest.mark.parametrize("path", sorted((FIXTURES_ROOT / "dimont").glob("*.xml")), ids=lambda path: path.name)
def test_dimont_fixture_site_scores_match_java_reference(path: Path):
    """Dimont fixture scores should match the Java parameter-tree semantics on both strands."""
    model = read_model(str(path), "dimont")
    rng = np.random.default_rng(sum(path.name.encode("utf-8")))

    for _ in range(20):
        sequence = rng.integers(0, 4, size=model.length, dtype=np.int8)
        site = make_sequence_batch([sequence])

        plus_score = float(row_values(scan_model(model, site, "+"), 0)[0])
        minus_score = float(row_values(scan_model(model, site, "-"), 0)[0])

        expected_plus = _reference_dimont_site_score(path, sequence)
        expected_minus = _reference_dimont_site_score(path, _reverse_complement_encoded(sequence))

        assert plus_score == pytest.approx(expected_plus, abs=1e-5)
        assert minus_score == pytest.approx(expected_minus, abs=1e-5)


@pytest.mark.parametrize("path", sorted((FIXTURES_ROOT / "slim").glob("*.xml")), ids=lambda path: path.name)
def test_slim_fixture_site_scores_match_java_reference(path: Path):
    """Slim fixture scores should match the Java higher-order mixture formula on both strands."""
    model = read_model(str(path), "slim")
    rng = np.random.default_rng(sum(path.name.encode("utf-8")))

    for _ in range(20):
        sequence = rng.integers(0, 4, size=model.length, dtype=np.int8)
        site = make_sequence_batch([sequence])

        plus_score = float(row_values(scan_model(model, site, "+"), 0)[0])
        minus_score = float(row_values(scan_model(model, site, "-"), 0)[0])

        expected_plus = _reference_slim_site_score(path, sequence)
        expected_minus = _reference_slim_site_score(path, _reverse_complement_encoded(sequence))

        assert plus_score == pytest.approx(expected_plus, abs=1e-5)
        assert minus_score == pytest.approx(expected_minus, abs=1e-5)


def test_read_model_bamm_defaults_to_max_order_and_derives_kmer():
    """BaMM loading should preserve the highest-order tensor unless an explicit order is requested."""
    myog_path = EXAMPLES_ROOT / "myog.ihbcp"
    _, myog_max_order, _ = parse_file_content(str(myog_path))

    default_order = read_model(str(myog_path), "bamm")
    second_order = read_model(str(myog_path), "bamm", order=2)
    foxa1 = read_model(str(FIXTURES_ROOT / "bamm" / "PEAKS036274_FOXA1_P35582_MACS2_motif_1.ihbcp"), "bamm")

    assert default_order.config["order"] == myog_max_order
    assert default_order.config["kmer"] == myog_max_order + 1
    assert default_order.representation.ndim == myog_max_order + 2
    assert second_order.config["kmer"] == 3
    assert second_order.config["order"] == 2
    assert second_order.representation.shape == (5, 5, 5, 14)
    assert default_order.representation.shape != second_order.representation.shape

    assert foxa1.config["kmer"] == foxa1.config["order"] + 1
    assert foxa1.config["order"] > 0
    assert foxa1.representation.ndim == foxa1.config["order"] + 2


def test_read_model_bamm_ignores_missing_background_and_uses_uniform_log_odds(tmp_path):
    """BaMM loading should no longer require a background file when using a uniform background."""
    motif_path = tmp_path / "toy.ihbcp"
    motif_path.write_text("0.25 0.25 0.25 0.25\n", encoding="utf-8")

    model = read_model(str(motif_path), "bamm")

    assert model.config["kmer"] == 1
    np.testing.assert_allclose(model.representation[:4, 0], np.zeros(4, dtype=np.float32), atol=1e-6)
