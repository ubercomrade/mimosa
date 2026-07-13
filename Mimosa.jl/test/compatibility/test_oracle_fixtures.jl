using Test
using Mimosa
using JSON3

# REPO_ROOT, EXAMPLES, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

@testset "pwm_parse_meme_pif4" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"); index=0)
    expected = read_npy(joinpath(FIXTURE_COMPAT, "pwm_parse_meme_pif4__pfm.npy"))
    @test pfm.frequencies == expected
    md = fixture_metadata("pwm_parse_meme_pif4")
    @test pfm.name == md["name"]
    @test size(pfm.frequencies) == tuple(md["shape"]...)
end

@testset "pwm_parse_pfm_pif4" begin
    pfm = read_pfm(joinpath(EXAMPLES, "pif4.pfm"))
    expected = read_npy(joinpath(FIXTURE_COMPAT, "pwm_parse_pfm_pif4__pfm.npy"))
    @test pfm.frequencies == expected
end

@testset "pwm_to_pwm_from_pif4" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"); index=0)
    pwm4 = pfm_to_pwm(pfm.frequencies)
    expected = read_npy(joinpath(FIXTURE_COMPAT, "pwm_to_pwm_from_pif4__pwm.npy"))
    @test pwm4 ≈ expected
end

@testset "pwm_reverse_complement_pif4" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"); index=0)
    pwm4 = pfm_to_pwm(pfm.frequencies)
    forward_expected = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_reverse_complement_pif4__forward.npy")
    )
    @test pwm4 ≈ forward_expected
    rc = reverse_complement(pwm4)
    reverse_expected = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_reverse_complement_pif4__reverse.npy")
    )
    @test rc ≈ reverse_expected
end

@testset "pwm_score_bounds_pif4" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    mn, mx = scorebounds(pwm)
    md = fixture_metadata("pwm_score_bounds_pif4")
    @test mn ≈ Float32(md["min_score"])
    @test mx ≈ Float32(md["max_score"])
end
