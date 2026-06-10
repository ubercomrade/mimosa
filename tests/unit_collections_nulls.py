# ruff: noqa: F403,F405

from tests.unit_support import *


def test_read_meme_many_and_read_models_load_multi_meme(tmp_path):
    """Collection loading should preserve all MEME motifs in file order."""
    meme_path = tmp_path / "two.meme"
    meme_path.write_text(
        """MEME version 4

ALPHABET= ACGT

MOTIF alpha
letter-probability matrix: alength= 4 w= 3 nsites= 10 E= 0
0.7 0.1 0.1 0.1
0.1 0.7 0.1 0.1
0.1 0.1 0.7 0.1

MOTIF beta
letter-probability matrix: alength= 4 w= 3 nsites= 10 E= 0
0.1 0.1 0.1 0.7
0.1 0.1 0.7 0.1
0.1 0.7 0.1 0.1
""",
        encoding="utf-8",
    )

    raw = read_meme_many(meme_path)
    models = read_models(meme_path, "pwm")

    assert [info[0] for _, info in raw] == ["alpha", "beta"]
    assert [model.name for model in models] == ["alpha", "beta"]


def test_read_models_directory_is_deterministic(tmp_path):
    """Directory collection loading should sort paths and keep stable model names."""
    first = tmp_path / "b_second.pfm"
    second = tmp_path / "a_first.pfm"
    first.write_text(">b_second\n0.7 0.1 0.1 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    second.write_text(">a_first\n0.1 0.7 0.1 0.1\n0.7 0.1 0.1 0.1\n", encoding="utf-8")

    models = read_models(tmp_path, "pwm", pattern="*.pfm")

    assert [model.name for model in models] == ["a_first", "b_second"]


def test_relation_parsers_exclude_self_and_group_matches(tmp_path):
    """Relation parsers should include only unrelated or explicitly included non-self pairs."""
    groups = tmp_path / "groups.tsv"
    groups.write_text("motif\tfamily\nq\tA\nt1\tA\nt2\tB\n", encoding="utf-8")
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("query\ttarget\tinclude\nq\tq\ttrue\nq\tt1\tfalse\nq\tt2\ttrue\n", encoding="utf-8")

    group_relations = parse_group_relations(groups, group_column="family", known_names={"q", "t1", "t2"})
    pair_relations = parse_pair_relations(pairs, known_names={"q", "t1", "t2"})

    assert group_relations["q"] == {"t2"}
    assert pair_relations["q"] == {"t2"}


def test_null_estimators_and_bh_qvalues_are_bounded_and_monotone():
    """Survival estimators and BH-FDR should return bounded upper-tail values."""
    estimator = fit_survival_estimator([0.1, 0.2, 0.3, 0.4])
    empirical = EmpiricalSurvivalEstimator([0.1, 0.2, 0.3, 0.4])

    assert 0.0 < estimator.sf(0.35) <= 1.0
    assert estimator.sf(0.2) >= estimator.sf(0.3)
    assert empirical.sf(0.4) == pytest.approx(2.0 / 5.0)
    assert empirical.sf(0.5) == pytest.approx(1.0 / 5.0)
    np.testing.assert_allclose(bh_qvalues([0.03, 0.01, 0.02]), [0.03, 0.03, 0.03])


def test_null_builder_and_annotation_add_significance_values():
    """Null builder should score included targets and annotation should add p/E/q values."""
    query = _make_shifted_core_pwm_model("q", 0)
    target_a = _make_shifted_core_pwm_model("u1", 1)
    target_b = _make_shifted_core_pwm_model("u2", 2)
    other_query = _make_shifted_core_pwm_model("q2", 3)
    config = create_comparator_config(metric="pcc")
    relations = {"q": {"q", "u1", "u2"}, "q2": {"u1", "u2"}}

    built = build_null_distributions(
        [query, target_a, target_b, other_query],
        relations,
        strategy="motif",
        config=config,
        min_null_targets=2,
    )
    distribution = built.null_distribution_file["distribution"]
    results = [
        ComparisonResult(
            query="q",
            target="u1",
            score=float(distribution["raw_null_scores"][0]),
            offset=0,
            orientation="++",
            metric="pcc",
        ),
        ComparisonResult(
            query="q",
            target="u2",
            score=float(distribution["raw_null_scores"][1]),
            offset=0,
            orientation="++",
            metric="pcc",
        ),
    ]
    annotated = annotate_results_with_nulls(
        results,
        null_distribution_file=built.null_distribution_file,
        query_model=query,
        effective_number_of_targets=2,
    )

    assert distribution["included_query_names"] == ["q", "q2"]
    assert distribution["included_target_names"] == ["u1", "u2"]
    assert distribution["n_null"] == 4
    assert len(distribution["included_pairs"]) == 4
    assert all("p-value" in result and "E-value" in result and "q-value" in result for result in annotated)


def test_build_null_request_from_args_runs_without_subprocess(tmp_path):
    """CLI build-null orchestration should be testable through request objects."""
    motif_dir = tmp_path / "motifs"
    motif_dir.mkdir()
    (motif_dir / "q.pfm").write_text(">q\n0.7 0.1 0.1 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    (motif_dir / "t1.pfm").write_text(">t1\n0.1 0.7 0.1 0.1\n0.7 0.1 0.1 0.1\n", encoding="utf-8")
    (motif_dir / "t2.pfm").write_text(">t2\n0.1 0.1 0.7 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    groups = tmp_path / "groups.tsv"
    groups.write_text("motif\tgroup\nq\tA\nt1\tB\nt2\tB\n", encoding="utf-8")
    output = tmp_path / "motif-pcc.null.joblib"
    args = SimpleNamespace(
        mode="build-null",
        motifs=motif_dir,
        model_type="pwm",
        pattern="*.pfm",
        groups=groups,
        pair_table=None,
        pair_matrix=None,
        name_column="motif",
        group_column="group",
        query_column="query",
        target_column="target",
        include_column="include",
        ignore_missing_relations=False,
        strategy="motif",
        metric="pcc",
        fasta=None,
        background=None,
        num_sequences=1000,
        seq_length=200,
        search_range=10,
        window_radius=10,
        realign_window=3,
        min_logfpr=None,
        pfm_mode=False,
        pfm_top_fraction=0.05,
        cache="off",
        cache_dir=".mimosa-cache",
        output=output,
        install_cache=False,
        strict=False,
        min_null_targets=2,
        jobs=None,
        seed=127,
    )

    request = build_null_request_from_args(args)
    summary = run_build_null_request(request)

    assert output.exists()
    assert request.relations["q"] == {"t1", "t2"}
    assert summary.to_dict()["null_distribution_file"] == str(output)
    assert summary.number_of_motifs == 3
    assert summary.number_of_queries_used == 1
    assert summary.total_comparisons_run == 2


def test_create_null_distribution_api_builds_from_in_memory_models(tmp_path):
    """Interactive null API should accept preloaded models and in-memory relations."""
    query = _make_shifted_core_pwm_model("q", 0)
    target_a = _make_shifted_core_pwm_model("u1", 1)
    target_b = _make_shifted_core_pwm_model("u2", 2)
    output = tmp_path / "interactive.null.joblib"

    request = create_null_distribution_config(
        [query, target_a, target_b],
        relations={"q": {"q", "u1", "u2"}},
        strategy="motif",
        metric="pcc",
        output=output,
        min_null_targets=2,
    )
    summary = run_null_distribution(request)

    assert output.exists()
    assert request.relations["q"] == {"u1", "u2"}
    assert summary.null_distribution_file == output
    assert summary.number_of_motifs == 3
    assert summary.number_of_queries_used == 1
    assert summary.total_comparisons_run == 2


def test_create_null_distribution_shortcut_builds_from_collection_path(tmp_path):
    """One-shot null API should load a collection path and group relations."""
    motif_dir = tmp_path / "motifs"
    motif_dir.mkdir()
    (motif_dir / "q.pfm").write_text(">q\n0.7 0.1 0.1 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    (motif_dir / "t1.pfm").write_text(">t1\n0.1 0.7 0.1 0.1\n0.7 0.1 0.1 0.1\n", encoding="utf-8")
    (motif_dir / "t2.pfm").write_text(">t2\n0.1 0.1 0.7 0.1\n0.1 0.7 0.1 0.1\n", encoding="utf-8")
    groups = tmp_path / "groups.tsv"
    groups.write_text("motif\tgroup\nq\tA\nt1\tB\nt2\tB\n", encoding="utf-8")
    output = tmp_path / "shortcut.null.joblib"

    summary = create_null_distribution(
        motif_dir,
        model_type="pwm",
        pattern="*.pfm",
        groups=groups,
        strategy="motif",
        metric="pcc",
        output=output,
        min_null_targets=2,
    )

    assert output.exists()
    assert summary.number_of_motifs == 3
    assert summary.number_of_queries_used == 1
    assert summary.total_comparisons_run == 2


def test_null_distribution_file_matching_rejects_incompatible_metric_but_not_query():
    """Pooled null distribution compatibility should not depend on the query fingerprint."""
    query = _make_shifted_core_pwm_model("q", 0)
    config = create_comparator_config(metric="pcc")
    built = build_null_distributions(
        [query, _make_shifted_core_pwm_model("u1", 1)],
        {"q": {"u1"}},
        strategy="motif",
        config=config,
    )

    incompatible = create_comparator_config(metric="cosine")
    other_query = _make_shifted_core_pwm_model("other", 0)

    assert is_null_distribution_file_compatible(
        built.null_distribution_file,
        strategy="motif",
        config=config,
        query_model=query,
    )
    assert not is_null_distribution_file_compatible(
        built.null_distribution_file,
        strategy="motif",
        config=incompatible,
        query_model=query,
    )
    assert is_null_distribution_file_compatible(
        built.null_distribution_file,
        strategy="motif",
        config=config,
        query_model=other_query,
    )


def test_environment_metadata_uses_distribution_name(monkeypatch):
    """Null metadata should query the installed distribution name."""
    calls = []

    def fake_version(distribution_name):
        calls.append(distribution_name)
        return "9.9.9"

    monkeypatch.setattr("mimosa.nulls.package_metadata.version", fake_version)

    metadata = environment_metadata(strategy="motif", config=create_comparator_config(metric="pcc"))

    assert calls == ["mimosa-tool"]
    assert metadata["package_version"] == "9.9.9"


def test_save_null_distribution_file_roundtrip(tmp_path):
    """Null distribution files should be saved as joblib payloads."""
    path = tmp_path / "null.joblib"
    null_distribution_file = {"metadata": {"format_version": 2}, "distribution": {"estimator_type": "ecdf"}}

    save_null_distribution_file(null_distribution_file, path)

    assert joblib.load(path) == null_distribution_file


if __name__ == "__main__":
    pytest.main([__file__])
