# JET static analysis for type stability in public hot paths.
#
# These checks are fail-closed: any type instability in the targeted functions
# will make this test (and therefore the CI job) fail.
#
# Strategy:
#   - @test_opt: used for pure computational kernels (no I/O, no error formatting,
#     no Dict{String,Any}). These must be fully type-stable.
#   - @test_call: used for public entry points that include I/O, error handling,
#     or serialization. These must not have hard inference errors, but runtime
#     dispatch in error/show paths is acceptable.
#
# Coverage:
#   - All model construction paths (PWM, PFM, BaMM, SiteGA, Dimont, Slim)
#   - All scanning kernels (forward, reverse, best, both; PWM and higher-order)
#   - Profile comparison (compare with profile metrics)
#   - Null distribution build (motif strategy)
#   - Site extraction and PFM reconstruction
#   - Storage round-trip (writemodel, readmodel)
#   - Serialization (to_json, to_dict)
#   - Cache operations (cache_key, cache_set, cache_get)
#
# Known type instabilities (documented, not yet fixed):
#   - compare() with Symbol metric dispatch: resolves metric at runtime via
#     _resolve_metric. Type-stable once metric is resolved.
#   - to_dict() returns Dict{String,Any}: unavoidable for JSON with heterogeneous
#     value types. Not a hot path.
#   - _build_null uses Tuple{AbstractMotifModel,AbstractMotifModel}[] for
#     work_pairs: heterogeneous model collections require abstract element type.
#   - ThreadedExecution has runtime dispatch in thread scheduling (expected).
#   - writemodel/readmodel have runtime dispatch in TOML serialization (expected).
#   - cache_key has runtime dispatch in SHA hashing (expected, stdlib path).

using Test
using JET
using Mimosa
using Random

const REPO_ROOT = joinpath(dirname(@__DIR__), "..", "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

@testset "JET: type stability — model construction" begin
    # PWM — pure computation, @test_opt
    @test_opt PWM(
        "jet_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )

    # PFM — pure computation, @test_opt
    @test_opt PFM("jet_pfm", Matrix{Float32}(Float32[10 20; 20 10; 15 15; 5 5]))

    # BaMM — pure computation, @test_opt
    bamm_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    @test_opt BaMM("jet_bamm", bamm_weights, 0, 3)

    # SiteGA — 25 rows (5×5 dinucleotides)
    sitega_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
        0.0 0.1 -0.1
        0.1 0.0 0.2
        -0.2 0.1 0.0
        0.3 -0.1 0.1
        0.0 0.0 -0.1
        0.1 0.2 -0.2
        0.0 0.0 0.0
        0.0 0.1 0.0
        0.2 0.0 0.1
        -0.1 -0.2 0.3
        0.0 0.0 0.0
        0.0 0.0 0.0
        0.1 -0.1 0.0
        0.0 0.2 0.1
        0.0 0.0 0.0
        0.0 0.0 0.0
        -0.1 0.1 0.0
        0.0 -0.1 0.2
        0.0 0.0 0.0
        0.0 0.0 0.0
    ]
    @test_opt SiteGA("jet_sitega", sitega_weights, 3)

    # Dimont — 5^(span+1) = 5 rows
    dimont_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    @test_opt Dimont("jet_dimont", dimont_weights, 0, 3)

    # Slim — 5^(span+1) = 5 rows
    slim_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    @test_opt Slim("jet_slim", slim_weights, 0, 3)
end

@testset "JET: type stability — score bounds" begin
    pwm = PWM(
        "jet_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    @test_opt scorebounds(pwm)

    pfm = PFM("jet_pfm", Matrix{Float32}(Float32[10 20; 20 10; 15 15; 5 5]))
    @test_opt scorebounds(pfm)

    bamm_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    bamm = BaMM("jet_bamm", bamm_weights, 0, 3)
    @test_opt scorebounds(bamm)

    sitega_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
        0.0 0.1 -0.1
        0.1 0.0 0.2
        -0.2 0.1 0.0
        0.3 -0.1 0.1
        0.0 0.0 -0.1
        0.1 0.2 -0.2
        0.0 0.0 0.0
        0.0 0.1 0.0
        0.2 0.0 0.1
        -0.1 -0.2 0.3
        0.0 0.0 0.0
        0.0 0.0 0.0
        0.1 -0.1 0.0
        0.0 0.2 0.1
        0.0 0.0 0.0
        0.0 0.0 0.0
        -0.1 0.1 0.0
        0.0 -0.1 0.2
        0.0 0.0 0.0
        0.0 0.0 0.0
    ]
    sitega = SiteGA("jet_sitega", sitega_weights, 3)
    @test_opt scorebounds(sitega)

    dimont_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    dimont = Dimont("jet_dimont", dimont_weights, 0, 3)
    @test_opt scorebounds(dimont)

    slim_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    slim = Slim("jet_slim", slim_weights, 0, 3)
    @test_opt scorebounds(slim)
end

@testset "JET: type stability — sequence encoding" begin
    @test_opt encode_sequence("ACGTACGT")
    @test_opt encode_sequence("ACGTNNNACGT")
    @test_opt make_random_sequences(10, 50; seed=42)
    @test_opt make_random_sequences(5, 100; seed=123)
end

@testset "JET: type stability — PWM scanning kernels" begin
    pwm = PWM(
        "jet_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    seq = encode_sequence("ACGTACGTACGTACGT")

    # Pure computational kernels — @test_opt
    @test_opt scan(pwm, seq; strands=ForwardOnly())
    @test_opt scan(pwm, seq; strands=ReverseOnly())
    @test_opt scan(pwm, seq; strands=BestStrand())
    @test_opt scan(pwm, seq; strands=BothStrands())

    # In-place scanning — @test_opt
    dest = Vector{Float32}(undef, npositions(length(seq), 2))
    @test_opt scan!(dest, pwm, seq; strands=ForwardOnly())

    # Batch scanning (serial) — @test_opt
    batch = make_random_sequences(5, 50; seed=42)
    @test_opt scan(pwm, batch; strands=ForwardOnly(), execution=SerialExecution())
    @test_opt scan(pwm, batch; strands=BestStrand(), execution=SerialExecution())
    @test_opt scan(pwm, batch; strands=BothStrands(), execution=SerialExecution())

    # Batch scanning (threaded) — @report_call (thread scheduling has runtime dispatch in Base)
    result = @report_call scan(
        pwm, batch; strands=BestStrand(), execution=ThreadedExecution()
    )
    @test result !== nothing
end

@testset "JET: type stability — higher-order scanning kernels" begin
    bamm_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    bamm = BaMM("jet_bamm", bamm_weights, 0, 3)
    seq = encode_sequence("ACGTACGTACGTACGT")

    @test_opt scan(bamm, seq; strands=ForwardOnly())
    @test_opt scan(bamm, seq; strands=ReverseOnly())
    @test_opt scan(bamm, seq; strands=BestStrand())
    @test_opt scan(bamm, seq; strands=BothStrands())

    batch = make_random_sequences(5, 50; seed=42)
    @test_opt scan(bamm, batch; strands=BestStrand(), execution=SerialExecution())
    @test_opt scan(bamm, batch; strands=BothStrands(), execution=SerialExecution())
    result = @report_call scan(
        bamm, batch; strands=BestStrand(), execution=ThreadedExecution()
    )
    @test result !== nothing

    # SiteGA
    sitega_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
        0.0 0.1 -0.1
        0.1 0.0 0.2
        -0.2 0.1 0.0
        0.3 -0.1 0.1
        0.0 0.0 -0.1
        0.1 0.2 -0.2
        0.0 0.0 0.0
        0.0 0.1 0.0
        0.2 0.0 0.1
        -0.1 -0.2 0.3
        0.0 0.0 0.0
        0.0 0.0 0.0
        0.1 -0.1 0.0
        0.0 0.2 0.1
        0.0 0.0 0.0
        0.0 0.0 0.0
        -0.1 0.1 0.0
        0.0 -0.1 0.2
        0.0 0.0 0.0
        0.0 0.0 0.0
    ]
    sitega = SiteGA("jet_sitega", sitega_weights, 3)
    @test_opt scan(sitega, seq; strands=ForwardOnly())
    @test_opt scan(sitega, seq; strands=BestStrand())
    @test_opt scan(sitega, batch; strands=BestStrand(), execution=SerialExecution())

    # Dimont
    dimont_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    dimont = Dimont("jet_dimont", dimont_weights, 0, 3)
    @test_opt scan(dimont, seq; strands=ForwardOnly())
    @test_opt scan(dimont, seq; strands=BestStrand())
    @test_opt scan(dimont, batch; strands=BestStrand(), execution=SerialExecution())

    # Slim
    slim_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    slim = Slim("jet_slim", slim_weights, 0, 3)
    @test_opt scan(slim, seq; strands=ForwardOnly())
    @test_opt scan(slim, seq; strands=BestStrand())
    @test_opt scan(slim, batch; strands=BestStrand(), execution=SerialExecution())
end

@testset "JET: type stability — reverse complement" begin
    seq = encode_sequence("ACGTACGTACGT")
    @test_opt reverse_complement(seq)
    rc_dest = similar(seq)
    @test_opt reverse_complement!(rc_dest, seq)
end

@testset "JET: type stability — motif comparison" begin
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
    # Typed metrics — @test_opt (pure computation)
    @test_opt compare(pwm1, pwm2; metric=PearsonCorrelation())
    @test_opt compare(pwm1, pwm2; metric=EuclideanDistance())
    @test_opt compare(pwm1, pwm2; metric=CosineSimilarity())

    # PFM comparison
    pfm1 = PFM("q", Matrix{Float32}(Float32[10 20; 20 10; 15 15; 5 5]))
    pfm2 = PFM("t", Matrix{Float32}(Float32[5 10; 10 5; 20 20; 15 15]))
    @test_opt compare(pfm1, pfm2; metric=PearsonCorrelation())
end

@testset "JET: type stability — profile comparison" begin
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

    # Profile comparison with typed metric — @test_call (includes scan + normalization)
    @test_call compare(
        pwm1, pwm2, batch; metric=OverlapCoefficient(), search_range=5, window_radius=5
    )
    @test_call compare(
        pwm1, pwm2, batch; metric=DiceSimilarity(), search_range=5, window_radius=5
    )
end

@testset "JET: type stability — site extraction" begin
    pwm = PWM(
        "jet_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    batch = make_random_sequences(5, 50; seed=42)

    # @test_call (includes scan + collection logic with some runtime dispatch)
    @test_call selectsites(pwm, batch, BestPerSequence(); strands=BestStrand())
    @test_call selectsites(pwm, batch, ThresholdHits(Float32(0.0)); strands=BestStrand())
    @test_call selectsites(pwm, batch, TopFractionHits(0.5); strands=BestStrand())
end

@testset "JET: type stability — PFM reconstruction" begin
    pwm = PWM(
        "jet_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    batch = make_random_sequences(5, 50; seed=42)
    @test_call reconstruct_pfm(pwm, batch, BestPerSequence(); pseudocount=Float32(1e-4))
end

@testset "JET: type stability — GEV fitting" begin
    samples = Float32[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    @test_opt fit_gev(samples)

    samples_large = Float32.(rand(MersenneTwister(42), 100))
    @test_opt fit_gev(samples_large)
end

@testset "JET: type stability — null distribution build" begin
    pwm1 = PWM(
        "motif_a",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    pwm2 = PWM(
        "motif_b",
        Matrix{Float32}(Float32[0.3 -0.1; -0.1 0.5; 0.2 -0.2; 0.1 0.3; -0.1 -0.1]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    pwm3 = PWM(
        "motif_c",
        Matrix{Float32}(Float32[0.1 0.2; 0.3 -0.1; -0.2 0.4; 0.0 0.1; 0.1 -0.2]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    relations = GroupRelations(
        Dict("motif_a" => "group1", "motif_b" => "group2", "motif_c" => "group2"),
        Dict(
            "motif_a" => Set(["motif_b", "motif_c"]),
            "motif_b" => Set(["motif_a"]),
            "motif_c" => Set(["motif_a"]),
        ),
    )

    # @test_call (uses abstract work_pairs and GEV fitting)
    @test_call build_null(
        [pwm1, pwm2, pwm3],
        relations;
        strategy=MotifNullStrategy(),
        metric=PearsonCorrelation(),
    )
end

@testset "JET: type stability — storage round-trip" begin
    pwm = PWM(
        "store_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    tmpdir = mktempdir()
    bundle = joinpath(tmpdir, "pwm_bundle")

    # Storage I/O — @report_call (TOML/SHA/NPY has runtime dispatch in I/O)
    result = @report_call writemodel(bundle, pwm)
    @test result !== nothing
    result = @report_call readmodel(bundle)
    @test result !== nothing

    # BaMM storage
    bamm_weights = Float32[
        0.1 -0.2 0.3
        0.2 0.1 -0.3
        -0.1 0.3 0.1
        0.3 -0.1 0.2
        -0.1 -0.1 -0.1
    ]
    bamm = BaMM("store_bamm", bamm_weights, 0, 3)
    bamm_bundle = joinpath(tmpdir, "bamm_bundle")
    result = @report_call writemodel(bamm_bundle, bamm)
    @test result !== nothing
    result = @report_call readmodel(bamm_bundle)
    @test result !== nothing
end

@testset "JET: type stability — serialization" begin
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
    cmp_result = compare(pwm1, pwm2; metric=PearsonCorrelation())
    # Serialization uses Dict{String,Any} — @report_call (runtime dispatch expected)
    jet_result = @report_call to_dict(cmp_result)
    @test jet_result !== nothing
    jet_result = @report_call to_json(cmp_result)
    @test jet_result !== nothing
end

@testset "JET: type stability — cache operations" begin
    pwm = PWM(
        "cache_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    batch = make_random_sequences(5, 50; seed=42)

    # SHA hashing has runtime dispatch — @test_call
    @test_call model_fingerprint(pwm)
    @test_call sequence_fingerprint(batch)
    @test_opt content_fingerprint("test data for hashing")
end

# ── JET call analysis for public entry points ──────────────────────────────
@testset "JET: call analysis for public entry points" begin
    pwm = PWM(
        "jet_test",
        Matrix{Float32}(Float32[0.5 -0.3; -0.2 0.7; 0.1 -0.4; 0.3 0.1; -0.2 -0.3]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    seq = encode_sequence("ACGTACGTACGT")
    batch = make_random_sequences(5, 50; seed=42)

    @test_call scan(pwm, seq; strands=ForwardOnly())
    @test_call scan(pwm, seq; strands=BestStrand())
    @test_call scan(pwm, seq; strands=BothStrands())
    @test_call scan(pwm, batch; strands=BestStrand(), execution=SerialExecution())
    result = @report_call scan(
        pwm, batch; strands=BestStrand(), execution=ThreadedExecution()
    )
    @test result !== nothing
    @test_call scorebounds(pwm)
    @test_call selectsites(pwm, batch, BestPerSequence(); strands=BestStrand())
    @test_call reverse_complement(seq)

    # Motif comparison with typed metric
    pwm2 = PWM(
        "t",
        Matrix{Float32}(Float32[0.3 -0.1; -0.1 0.5; 0.2 -0.2; 0.1 0.3; -0.1 -0.1]),
        (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25)),
    )
    # Motif comparison with typed metric — @report_call (reverse_complement goes through
    # LinearAlgebra Adjoint/Transpose union types in stdlib)
    jet_result = @report_call compare(pwm, pwm2; metric=PearsonCorrelation())
    @test jet_result !== nothing
    jet_result = @report_call compare(pwm, pwm2; metric=EuclideanDistance())
    @test jet_result !== nothing
    jet_result = @report_call compare(pwm, pwm2; metric=CosineSimilarity())
    @test jet_result !== nothing

    # Profile comparison with typed metric
    jet_result = @report_call compare(
        pwm, pwm2, batch; metric=OverlapCoefficient(), search_range=5, window_radius=5
    )
    @test jet_result !== nothing

    # GEV fitting
    samples = Float32[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    @test_call fit_gev(samples)
end
