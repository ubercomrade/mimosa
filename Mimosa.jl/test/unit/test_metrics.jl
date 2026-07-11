using Test
using Mimosa

@testset "metric_name and parse_metric" begin
    @test metric_name(PearsonCorrelation()) == "pcc"
    @test metric_name(EuclideanDistance()) == "ed"
    @test metric_name(CosineSimilarity()) == "cosine"
    @test parse_metric("pcc") isa PearsonCorrelation
    @test parse_metric("ed") isa EuclideanDistance
    @test parse_metric("cosine") isa CosineSimilarity
    @test_throws ArgumentError parse_metric("nope")
end

@testset "PCC column metric" begin
    x = Float32[1.0 0.0; 0.0 1.0]
    y = Float32[0.0 1.0; 1.0 0.0]
    # Perfect anti-correlation per column → pcc = -1
    s = Mimosa.score_columns(PearsonCorrelation(), x, y)
    @test s ≈ -1.0f0
    # Identical → pcc = 1
    @test Mimosa.score_columns(PearsonCorrelation(), x, x) ≈ 1.0f0
end

@testset "ED column metric" begin
    x = Float32[0.0 1.0; 0.0 0.0; 0.0 0.0; 1.0 0.0]
    @test Mimosa.score_columns(EuclideanDistance(), x, x) ≈ 0.0f0
    y = Float32[1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 1.0]
    # Each column: sqrt(1+0+0+1)=sqrt(2); 2 cols → -2*sqrt(2)/2 = -sqrt(2)
    @test Mimosa.score_columns(EuclideanDistance(), x, y) ≈ -sqrt(2.0f0)
end

@testset "Cosine column metric" begin
    x = Float32[0.0 1.0; 0.0 0.0; 0.0 0.0; 1.0 0.0]
    @test Mimosa.score_columns(CosineSimilarity(), x, x) ≈ 1.0f0
    # Orthogonal columns → 0
    y = Float32[1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 1.0]
    @test Mimosa.score_columns(CosineSimilarity(), x, y) ≈ 0.0f0
end

@testset "zero variance PCC returns 0" begin
    x = Float32[0.25 0.25; 0.25 0.25; 0.25 0.25; 0.25 0.25]
    @test Mimosa.score_columns(PearsonCorrelation(), x, x) == 0.0f0
end
