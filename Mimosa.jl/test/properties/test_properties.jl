using Test
using Mimosa

const REPO_ROOT = joinpath(dirname(dirname(@__DIR__)), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

@testset "reverse_complement involution (PWM)" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    rc = reverse_complement(pwm)
    @test reverse_complement(rc) == pwm
end

@testset "reverse_complement involution (PFM)" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"))
    rc = reverse_complement(pfm)
    @test reverse_complement(rc) == pfm
end

@testset "identical motif comparison gives pcc 1.0" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    @test compare(pwm, pwm; metric="pcc").score ≈ 1.0f0
end

@testset "non-! functions do not mutate inputs" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    copy_w = copy(pwm.weights)
    compare(pwm, pwm; metric="pcc")
    scorebounds(pwm)
    reverse_complement(pwm)
    @test pwm.weights == copy_w
end

@testset "comparison is deterministic" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    r1 = compare(pwm, pwm; metric="pcc")
    r2 = compare(pwm, pwm; metric="pcc")
    @test r1.score == r2.score
    @test r1.offset == r2.offset
    @test r1.orientation == r2.orientation
end

@testset "score bounds are consistent" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    mn, mx = scorebounds(pwm)
    @test mn <= mx
    @test isfinite(mn) && isfinite(mx)
end

@testset "orientation labels are valid" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "gata2.meme"))
    for m in ("pcc", "ed", "cosine")
        r = compare(pwm1, pwm2; metric=m)
        @test r.orientation in ("++", "+-", "-+", "--")
    end
end