using Test
using Mimosa

const _NULL_BG = (0.25f0, 0.25f0, 0.25f0, 0.25f0)

@testset "profile-only null configuration" begin
    config = NullBuildConfig(metric=:co, min_null_targets=1)
    @test config.metric isa OverlapCoefficient
    @test config.min_null_targets == 1
    @test_throws ArgumentError NullBuildConfig(metric=:not_a_profile_metric)
    @test_throws ArgumentError NullBuildConfig(metric=:co, min_null_targets=0)
end

@testset "profile null build" begin
    w1 = Float32[0.8 0.1; 0.1 0.8; 0.05 0.05; 0.05 0.05; 0.0 0.0]
    w2 = Float32[0.1 0.8; 0.8 0.1; 0.05 0.05; 0.05 0.05; 0.0 0.0]
    models = [PWM("m1", w1, _NULL_BG), PWM("m2", w2, _NULL_BG)]
    relations = GroupRelations(
        Dict("m1" => "A", "m2" => "B"),
        Dict("m1" => Set(["m2"]), "m2" => Set(["m1"])),
    )
    sequences = EncodedSequenceBatch([
        encode_sequence("ACGTACGT"), encode_sequence("TGCATGCA")
    ])
    result = build_null(models, relations; sequences=sequences, metric=:co)
    @test result.distribution.strategy == "profile"
    @test result.distribution.metric == "co"
    @test result.distribution.sequence_fingerprint == sequence_fingerprint(sequences)
    @test result.total_comparisons == 2
end
