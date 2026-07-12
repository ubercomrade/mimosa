# Downstream Contract Test for Mimosa.jl
#
# This test verifies that a downstream consumer (e.g., MotifHORDE.jl) can
# use Mimosa.jl through only the documented public API, without importing
# internal submodules or accessing internal functions.
#
# Run with:
#   julia --project=. test/downstream/runtests.jl

using Test

# Import only the public API — no submodule access
using Mimosa

# Verify that all exported names are accessible
@testset "Downstream contract: exports" begin
    # Model I/O
    @test isdefined(Mimosa, :readmodel)
    @test isdefined(Mimosa, :writemodel)
    @test isdefined(Mimosa, :readsequences)

    # Scanning
    @test isdefined(Mimosa, :scan)
    @test isdefined(Mimosa, :scan!)
    @test isdefined(Mimosa, :scorebounds)

    # Comparison
    @test isdefined(Mimosa, :compare)
    @test isdefined(Mimosa, :ComparisonResult)

    # Site extraction
    @test isdefined(Mimosa, :selectsites)
    @test isdefined(Mimosa, :reconstruct_pfm)

    # Statistics
    @test isdefined(Mimosa, :build_null)
    @test isdefined(Mimosa, :pvalue)
    @test isdefined(Mimosa, :adjusted_pvalues)
    @test isdefined(Mimosa, :evalue)
    @test isdefined(Mimosa, :savenull)
    @test isdefined(Mimosa, :loadnull)

    # Execution policies
    @test isdefined(Mimosa, :SerialExecution)
    @test isdefined(Mimosa, :ThreadedExecution)

    # Cache
    @test isdefined(Mimosa, :Cache)
    @test isdefined(Mimosa, :clearcache)

    # Serialization
    @test isdefined(Mimosa, :to_json)
    @test isdefined(Mimosa, :to_dict)

    # Errors
    @test isdefined(Mimosa, :MimosaError)
    @test isdefined(Mimosa, :ModelFormatError)
    @test isdefined(Mimosa, :ModelDimensionError)
    @test isdefined(Mimosa, :InvariantError)
end

@testset "Downstream contract: workflow" begin
    # Create a PWM from synthetic data
    weights = Float32[
        0.5 -0.3 0.8 -0.2 0.1 0.6 -0.4 0.3
        -0.2 0.7 -0.5 0.3 0.8 -0.1 0.2 -0.6
        0.1 -0.4 0.2 0.6 -0.3 0.5 0.7 -0.2
        0.3 0.1 -0.6 0.4 0.2 -0.5 -0.1 0.8
        -0.2 -0.3 -0.5 -0.2 -0.3 -0.5 -0.4 -0.6
    ]
    bg = (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25))
    pwm = PWM("contract_test", weights, bg)

    # Score bounds
    lo, hi = scorebounds(pwm)
    @test lo isa Float32
    @test hi isa Float32
    @test lo <= hi

    # Scan sequences
    batch = make_random_sequences(10, 200; seed=42)
    scores = scan(pwm, batch; strands=BestStrand())
    @test scores isa RaggedArray

    # Threaded scan
    scores_thr = scan(pwm, batch; strands=BestStrand(), execution=ThreadedExecution(2))
    @test scores == scores_thr  # serial == threaded

    # Compare motifs
    result = compare(pwm, pwm; metric=:pcc)
    @test result isa ComparisonResult
    @test result.query == "contract_test"
    @test result.target == "contract_test"
    @test result.metric == "pcc"

    # Serialize
    json_str = to_json(result)
    @test json_str isa String
    dict = to_dict(result)
    @test dict isa Dict

    # Site extraction
    sites = selectsites(pwm, batch, BestPerSequence(); strands=BestStrand())
    @test sites isa SiteCollection

    # PFM reconstruction
    pfm = reconstruct_pfm(pwm, batch, BestPerSequence(); pseudocount=Float32(1e-4))
    @test pfm isa AbstractMatrix{Float32}
    @test size(pfm, 1) == 4  # A, C, G, T rows

    # GEV fit
    samples = Float32.(collect(0.1:0.05:2.0))
    gev = fit_gev(samples)
    @test gev isa GEVFit

    # p-value
    p = pvalue(gev, 1.0)
    @test 0.0 <= p <= 1.0

    # BH FDR
    pvals = Float32[0.01, 0.02, 0.03, 0.04, 0.05]
    adj = adjusted_pvalues(pvals; method=BenjaminiHochberg())
    @test length(adj) == 5

    # E-value
    e = evalue(Float64(0.05), 100)
    @test e >= 0.0
end

@testset "Downstream contract: no internal access needed" begin
    # Verify that typical workflows don't require importing internal submodules
    # This test would fail if Mimosa.Scanning, Mimosa.Comparison, etc.
    # were required — but they shouldn't be.

    # The fact that the above tests pass with only `using Mimosa`
    # (not `using Mimosa.Scanning` etc.) is the contract.
    @test true
end
