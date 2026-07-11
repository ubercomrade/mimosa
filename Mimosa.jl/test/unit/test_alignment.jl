using Test
using Mimosa

const REPO_ROOT = joinpath(dirname(dirname(@__DIR__)), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

@testset "align_motif_matrices self-alignment offset 0" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"))
    pwm = pwm_from_pfm(pfm)
    fwd, rev = Mimosa.prepare_motif(pwm.weights)
    score, offset = Mimosa.align_motif_matrices(fwd, fwd, PearsonCorrelation())
    @test offset == 0
    @test score ≈ 1.0f0
end

@testset "compare PWM vs PWM self pcc" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="pcc")
    @test result.query == "pwm_model"
    @test result.target == "pwm_model"
    @test result.metric == "pcc"
    @test result.orientation == "++"
    @test result.offset == 0
    @test result.score ≈ 1.0f0
end

@testset "compare self ed returns 0" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="ed")
    @test result.orientation == "++"
    @test result.offset == 0
    @test result.score ≈ 0.0f0 atol = 1e-5
end

@testset "compare self cosine returns 1" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="cosine")
    @test result.orientation == "++"
    @test result.offset == 0
    @test result.score ≈ 1.0f0
end

@testset "compare accepts typed metric" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric=PearsonCorrelation())
    @test result.metric == "pcc"
end

@testset "select_best tie-breaking by orientation rank" begin
    o1 = Mimosa.Orientation("++", 0)
    o2 = Mimosa.Orientation("+-", 1)
    o3 = Mimosa.Orientation("-+", 2)
    c1 = Mimosa.MotifCandidate(o1, 0, 0.5f0)
    c2 = Mimosa.MotifCandidate(o2, 1, 0.5f0)
    c3 = Mimosa.MotifCandidate(o3, -1, 0.5f0)
    best = Mimosa.select_best([c1, c2, c3])
    @test best.orientation.label == "++"
end

@testset "compare does not mutate inputs" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    weights_copy = copy(pwm.weights)
    compare(pwm, pwm; metric="pcc")
    @test pwm.weights == weights_copy
end
