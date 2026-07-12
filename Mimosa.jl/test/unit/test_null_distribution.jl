using Test
using Mimosa

@testset "Null distribution build" begin
    # Create simple PWM models for testing (5 rows: A,C,G,T,N)
    weights1 = Float32[
        0.1 0.9 0.1 0.9
        0.1 0.0 0.8 0.0
        0.7 0.0 0.1 0.0
        0.1 0.1 0.0 0.1
        0.0 0.0 0.0 0.0
    ]
    weights2 = Float32[
        0.8 0.2 0.7 0.1
        0.1 0.1 0.2 0.8
        0.0 0.5 0.1 0.0
        0.1 0.2 0.0 0.1
        0.0 0.0 0.0 0.0
    ]
    weights3 = Float32[
        0.3 0.7 0.2 0.8
        0.2 0.1 0.6 0.1
        0.4 0.1 0.1 0.0
        0.1 0.1 0.1 0.1
        0.0 0.0 0.0 0.0
    ]
    bg = (0.25f0, 0.25f0, 0.25f0, 0.25f0)
    m1 = PWM("m1", weights1, bg)
    m2 = PWM("m2", weights2, bg)
    m3 = PWM("m3", weights3, bg)

    # Create group relations: m1 in group A, m2 in group B, m3 in group C
    relations = GroupRelations(
        Dict("m1" => "A", "m2" => "B", "m3" => "C"),
        Dict(
            "m1" => Set(["m2", "m3"]), "m2" => Set(["m1", "m3"]), "m3" => Set(["m1", "m2"])
        ),
    )

    result = build_null([m1, m2, m3], relations; strategy="motif", metric=:pcc)

    @test result isa NullBuildResult
    @test result.total_comparisons == 6  # 3 models × 2 targets each
    dist = result.distribution
    @test dist isa NullDistribution
    @test dist.n_null == 6
    @test dist.n_queries == 3
    @test length(dist.raw_scores) == 6
    @test length(dist.pairs) == 6
    @test isempty(dist.skipped)
    @test dist.fit isa GEVFit
    @test dist.strategy == "motif"
    @test dist.metric == "pcc"
end

@testset "Null build with skipped queries" begin
    weights = Float32[
        0.1 0.9 0.1 0.9
        0.1 0.0 0.8 0.0
        0.7 0.0 0.1 0.0
        0.1 0.1 0.0 0.1
        0.0 0.0 0.0 0.0
    ]
    bg = (0.25f0, 0.25f0, 0.25f0, 0.25f0)
    m1 = PWM("m1", weights, bg)
    m2 = PWM("m2", weights, bg)

    # m1 in group A, m2 in group B — both have 1 target each
    relations = GroupRelations(
        Dict("m1" => "A", "m2" => "B"), Dict("m1" => Set(["m2"]), "m2" => Set(["m1"]))
    )

    # min_null_targets = 2 → both skipped, no comparisons
    @test_throws ArgumentError build_null([m1, m2], relations; min_null_targets=2)

    # strict = true should also throw (when a query is skipped)
    @test_throws ArgumentError build_null(
        [m1, m2], relations; min_null_targets=2, strict=true
    )
end

@testset "Annotate results" begin
    # Create a null distribution with known GEV
    gev = GEVFit(0.0, 0.0, 1.0, true, 10, -100.0)
    dist = NullDistribution(
        "motif",
        "pcc",
        gev,
        Float64[1.0, 2.0, 3.0, 4.0, 5.0],
        NullPair[],
        5,
        1,
        [],
        nothing,
        nothing,
        "none",
        "none",
    )

    # Create comparison results
    r1 = ComparisonResult("q1", "t1", 1.0f0, 0, "++", "pcc", 0)
    r2 = ComparisonResult("q1", "t2", 2.0f0, 0, "++", "pcc", 0)
    r3 = ComparisonResult("q1", "t3", 0.0f0, 0, "++", "pcc", 0)

    annotated = annotate_results([r1, r2, r3], dist; effective_number_of_targets=3)

    @test length(annotated) == 3
    @test all(a isa AnnotatedResult for a in annotated)

    # p-values should be valid
    for a in annotated
        @test 0.0 ≤ a.p_value ≤ 1.0
        @test a.e_value ≈ a.p_value * 3
        @test a.null_n == 5
        @test a.null_estimator == "genextreme"
        @test a.null_id !== nothing
    end

    # Higher score → lower p-value
    @test annotated[1].p_value < annotated[3].p_value

    # Adjusted p-values: with 3 p-values
    @test all(0.0 .<= [a.adj_p_value for a in annotated] .<= 1.0)
end

@testset "Annotate with failed GEV" begin
    fail = GEVFitFailure("test failure", 3, 0)
    dist = NullDistribution(
        "motif",
        "pcc",
        fail,
        [1.0, 2.0, 3.0],
        NullPair[],
        3,
        1,
        [],
        nothing,
        nothing,
        "none",
        "none",
    )

    r = ComparisonResult("q", "t", 1.0f0, 0, "++", "pcc", 0)
    @test_throws ArgumentError annotate_results([r], dist)
end
