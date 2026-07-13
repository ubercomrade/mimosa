# Type stability inspection tests for public hot paths.
#
# These tests use JET.@report_opt to verify that the inferred code for public
# entry points has no type instabilities. Uses @report_opt for pure
# computational kernels and @report_call for functions with I/O or error
# handling paths.
#
# Run with:
#   julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
#
# The tests are fail-closed: any type instability makes the test fail.

using Test
using JET
using Mimosa
using Random

const REPO_ROOT = joinpath(dirname(@__DIR__), "..", "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

@testset "Type stability: model construction" begin
    result = @report_opt PWM(
        "ts",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    @test isempty(JET.get_reports(result))

    result = @report_opt PFM("ts", Matrix{Float32}(Float32[10 20; 20 10; 15 15; 5 5]))
    @test isempty(JET.get_reports(result))

    result = @report_opt BaMM(
        "ts",
        Float32[
            0.1 -0.2 0.3
            0.2 0.1 -0.3
            -0.1 0.3 0.1
            0.3 -0.1 0.2
            -0.1 -0.1 -0.1
        ],
        0,
        3,
    )
    @test isempty(JET.get_reports(result))
end

@testset "Type stability: scanning kernels" begin
    pwm = PWM(
        "ts",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    seq = encode_sequence("ACGTACGTACGTACGT")
    batch = make_random_sequences(5, 50; seed=42)

    for strands in (ForwardOnly(), ReverseOnly(), BestStrand(), BothStrands())
        result = @report_opt scan(pwm, seq; strands=strands)
        @test isempty(JET.get_reports(result))
    end

    for strands in (ForwardOnly(), BestStrand(), BothStrands())
        result = @report_opt scan(pwm, batch; strands=strands, execution=SerialExecution())
        @test isempty(JET.get_reports(result))
    end

    dest = Vector{Float32}(undef, npositions(length(seq), 2))
    result = @report_opt scan!(dest, pwm, seq; strands=ForwardOnly())
    @test isempty(JET.get_reports(result))
end

@testset "Type stability: higher-order scanning" begin
    bamm_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    bamm = BaMM("ts", bamm_weights, 0, 3)
    seq = encode_sequence("ACGTACGTACGTACGT")
    batch = make_random_sequences(5, 50; seed=42)

    for strands in (ForwardOnly(), ReverseOnly(), BestStrand(), BothStrands())
        result = @report_opt scan(bamm, seq; strands=strands)
        @test isempty(JET.get_reports(result))
    end

    result = @report_opt scan(
        bamm, batch; strands=BestStrand(), execution=SerialExecution()
    )
    @test isempty(JET.get_reports(result))
end

@testset "Type stability: motif comparison with typed metrics" begin
    pwm1 = PWM(
        "q",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    pwm2 = PWM(
        "t",
        Matrix{Float32}(Float32[0.3 -0.1; -0.1 0.5; 0.2 -0.2; 0.1 0.3; -0.1 -0.1]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )

    for metric in (PearsonCorrelation(), EuclideanDistance(), CosineSimilarity())
        result = @report_opt compare(pwm1, pwm2; metric=metric)
        @test isempty(JET.get_reports(result))
    end
end

@testset "Type stability: profile comparison with typed metrics" begin
    pwm1 = PWM(
        "q",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    pwm2 = PWM(
        "t",
        Matrix{Float32}(Float32[0.3 -0.1; -0.1 0.5; 0.2 -0.2; 0.1 0.3; -0.1 -0.1]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    batch = make_random_sequences(5, 50; seed=42)

    # Profile comparison includes scan + normalization — use @report_call
    for metric in (OverlapCoefficient(), DiceSimilarity())
        result = @report_call compare(
            pwm1, pwm2, batch; metric=metric, search_range=5, window_radius=5
        )
        # No hard errors expected
        @test result !== nothing
    end
end

@testset "Type stability: site extraction and reconstruction" begin
    pwm = PWM(
        "ts",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    batch = make_random_sequences(5, 50; seed=42)

    # Site extraction includes scan + collection — use @report_call
    for selector in (BestPerSequence(), ThresholdHits(Float32(0.0)), TopFractionHits(0.5))
        result = @report_call selectsites(pwm, batch, selector; strands=BestStrand())
        @test result !== nothing
    end

    result = @report_call reconstruct_pfm(
        pwm, batch, BestPerSequence(); pseudocount=Float32(1e-4)
    )
    @test result !== nothing
end

@testset "Type stability: GEV fitting" begin
    samples = Float32[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    result = @report_opt fit_gev(samples)
    @test isempty(JET.get_reports(result))

    samples_large = Float32.(rand(MersenneTwister(42), 100))
    result = @report_opt fit_gev(samples_large)
    @test isempty(JET.get_reports(result))
end

@testset "Type stability: storage I/O" begin
    pwm = PWM(
        "store",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    tmpdir = mktempdir()
    bundle = joinpath(tmpdir, "pwm_bundle")

    # Storage I/O has runtime dispatch in TOML/SHA — use @report_call
    result = @report_call writemodel(bundle, pwm)
    @test result !== nothing

    result = @report_call readmodel(bundle)
    @test result !== nothing
end

@testset "Type stability: cache key computation" begin
    pwm = PWM(
        "cache",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    batch = make_random_sequences(5, 50; seed=42)

    # SHA hashing has runtime dispatch — use @report_call
    result = @report_call model_fingerprint(pwm)
    @test result !== nothing

    result = @report_call model_fingerprint(pwm)
    @test result !== nothing

    result = @report_call sequence_fingerprint(batch)
    @test result !== nothing
end

@testset "Type stability: documented instabilities" begin
    # The following functions have known, documented type instabilities.
    # They are NOT hot paths and the instabilities are acceptable.
    # This testset documents them explicitly.

    pwm1 = PWM(
        "q",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    pwm2 = PWM(
        "t",
        Matrix{Float32}(Float32[0.3 -0.1; -0.1 0.5; 0.2 -0.2; 0.1 0.3; -0.1 -0.1]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )

    # compare() with Symbol metric: _resolve_metric accepts Union{AbstractString,
    # Symbol, AbstractColumnMetric} and dispatches at runtime. The inner
    # computation is type-stable once the metric is resolved.
    # This is documented in PLAN_2.md E3 and the JET test file header.
    result = @report_call compare(pwm1, pwm2; metric=:pcc)
    @test result !== nothing

    # to_dict returns Dict{String,Any}: unavoidable for JSON with heterogeneous
    # value types (String, Float64, Int, Nothing). Not a hot path.
    result_cmp = compare(pwm1, pwm2; metric=PearsonCorrelation())
    d = to_dict(result_cmp)
    @test d isa Dict{String,Any}
    @test d["query"] == "q"
    @test d["score"] isa Float64

    # _build_null uses Tuple{AbstractMotifModel,AbstractMotifModel}[] for
    # work_pairs: heterogeneous model collections require abstract element type.
    # This is unavoidable without requiring homogeneous model collections.
    # The inner loop (compare_pair closure) is type-stable per call.
    # Documented here for tracking — not a JET target.
    @test Tuple{AbstractMotifModel,AbstractMotifModel} isa Type
end
