using Test
using Mimosa
using JET

@testset "JET profile entry point" begin
    profile = ScoreProfile("profile", build_ragged([Float32[0.1, 0.8, 0.2, 0.7]]))
    target = ScoreProfile("target", build_ragged([Float32[0.2, 0.7, 0.3, 0.6]]))
    @test_opt compare(profile, target; metric=OverlapCoefficient())
end
