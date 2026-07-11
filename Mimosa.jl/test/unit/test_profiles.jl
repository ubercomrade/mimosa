using Test
using Mimosa

@testset "LogTailTable fit" begin
    # Empty input
    t = fit(EmpiricalLogTail(), Float32[])
    @test length(t.scores) == 1
    @test t.scores[1] == 0.0f0
    @test t.log_tail[1] == 0.0f0

    # Single element
    t = fit(EmpiricalLogTail(), Float32[5.0])
    @test t.scores == [5.0f0]
    @test t.log_tail ≈ [-0.0f0] atol = 1e-6

    # Multiple elements
    t = fit(EmpiricalLogTail(), Float32[1.0, 2.0, 3.0, 1.0, 2.0, 1.0])
    # Sorted descending: [3,2,2,1,1,1]
    # Unique: [3,2,1], counts: [1,2,3]
    # Cum: [1,3,6], tail_prob: [1/6,3/6,6/6]
    # -log10: [0.778,0.301,0]
    @test t.scores == [3.0f0, 2.0f0, 1.0f0]
    @test t.log_tail[1] ≈ Float32(-log10(1.0 / 6.0)) atol = 1e-5
    @test t.log_tail[2] ≈ Float32(-log10(3.0 / 6.0)) atol = 1e-5
    @test t.log_tail[3] ≈ Float32(-log10(6.0 / 6.0)) atol = 1e-5
end

@testset "LogTailTable lookup" begin
    t = fit(EmpiricalLogTail(), Float32[1.0, 2.0, 3.0, 1.0, 2.0, 1.0])
    # scores descending: [3,2,1], log_tail: [0.778,0.301,0]

    # Target >= largest → index 1
    @test lookup_score(t, 3.0f0) ≈ t.log_tail[1]
    @test lookup_score(t, 5.0f0) ≈ t.log_tail[1]

    # Target <= smallest → index 3
    @test lookup_score(t, 1.0f0) ≈ t.log_tail[3]
    @test lookup_score(t, 0.0f0) ≈ t.log_tail[3]

    # Target between unique scores
    # descending scores [3,2,1]: _lower_bound_desc finds first score <= target
    @test lookup_score(t, 2.5f0) ≈ t.log_tail[2]  # 2.5 → scores[2]=2 <= 2.5
    @test lookup_score(t, 2.0f0) ≈ t.log_tail[2]  # 2.0 → scores[2]=2 <= 2.0
    @test lookup_score(t, 1.5f0) ≈ t.log_tail[3]  # 1.5 → scores[3]=1 <= 1.5
end

@testset "transform_scores" begin
    t = fit(EmpiricalLogTail(), Float32[1.0, 2.0, 3.0])
    # scores: [3,2,1], log_tail: [-log10(1/3), -log10(2/3), -log10(3/3)]
    # = [0.477, 0.176, 0]

    rag = build_ragged([Float32[3.0, 2.0, 1.0], Float32[2.0, 1.0]])
    transformed = transform_scores(t, rag)

    @test nrows(transformed) == 2
    @test rowlength(transformed, 1) == 3
    @test rowlength(transformed, 2) == 2
    @test row(transformed, 1) ≈ [t.log_tail[1], t.log_tail[2], t.log_tail[3]]
    @test row(transformed, 2) ≈ [t.log_tail[2], t.log_tail[3]]
end

@testset "normalize_bundle" begin
    # Build a simple strand bundle
    rag = build_ragged([Float32[3.0, 1.0], Float32[2.0, 2.0]])
    bundle = StrandPair(rag, rag)

    flat = flatten_bundle(bundle)
    @test length(flat) == 8  # 4 elements × 2 strands

    t = fit(EmpiricalLogTail(), flat)
    normed = normalize_bundle(t, bundle)

    # Both strands should be transformed
    @test nrows(normed.forward) == 2
    @test nrows(normed.reverse) == 2
    @test row(normed.forward, 1) ≈ row(normed.reverse, 1)
end

@testset "AnchorCSR" begin
    # Build anchors: row 1 has positions [3, 5], row 2 has position [1]
    rows = [1, 1, 2]
    positions = [3, 5, 1]
    csr = build_anchor_csr(rows, positions, 3)

    @test csr.offsets == [1, 3, 4, 4]  # row 1: [1:2], row 2: [3:3], row 3: empty
    @test csr.positions == [3, 5, 1]  # stable sort preserves order

    # Empty case
    csr_empty = build_anchor_csr(Int[], Int[], 2)
    @test isempty(csr_empty)
    @test csr_empty.offsets == [1, 1, 1]
end

@testset "collect_best_anchors" begin
    rag = build_ragged([
        Float32[0.1, 0.5, 0.3],  # best at position 2
        Float32[0.7, 0.2, 0.9, 0.1],  # best at position 3
        Float32[],  # empty row
    ])

    rows, positions = collect_best_anchors(rag)
    @test rows == [1, 2]
    @test positions == [2, 3]
end

@testset "collect_threshold_anchors" begin
    rag = build_ragged([Float32[0.1, 0.5, 0.3], Float32[0.7, 0.2, 0.9, 0.1]])

    rows, positions = collect_threshold_anchors(rag, 0.4f0)
    @test rows == [1, 2, 2]
    @test positions == [2, 1, 3]
end

@testset "ProfileConfig defaults" begin
    config = ProfileConfig()
    @test config.metric isa OverlapCoefficient
    @test config.search_range == 10
    @test config.window_radius == 10
    @test config.realign_window == 3
    @test config.min_logfpr == 0.0f0

    config2 = ProfileConfig(metric=DiceSimilarity(), search_range=5, window_radius=3)
    @test config2.metric isa DiceSimilarity
    @test config2.search_range == 5
    @test config2.window_radius == 3
end
