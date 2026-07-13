# Downstream Contract Test for Mimosa.jl
#
# This test verifies that a downstream consumer (e.g., MotifHORDE.jl) can
# use Mimosa.jl through only the documented public API, without importing
# internal submodules or accessing internal functions.
#
# Run with:
#   julia --project=test/downstream test/downstream/runtests.jl
#
# The test environment is independent of the main test suite: it does not
# include any shared test helpers, NPY readers, or fixture metadata.

using Pkg
# Ensure Mimosa is available from the local package root.
Pkg.develop(; path=dirname(dirname(@__DIR__)))

using Test
using Mimosa

# Path to example data files (repo-level, not test internals)
const REPO_ROOT = dirname(dirname(dirname(@__DIR__)))
const EXAMPLES = joinpath(REPO_ROOT, "examples")

# ---------------------------------------------------------------------------
# Verify that all exported names are accessible
# ---------------------------------------------------------------------------
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

    # Annotation
    @test isdefined(Mimosa, :annotate_results)
    @test isdefined(Mimosa, :AnnotatedResult)
    @test isdefined(Mimosa, :ANNOTATED_RESULT_SCHEMA_VERSION)

    # Null distribution types
    @test isdefined(Mimosa, :NullDistribution)
    @test isdefined(Mimosa, :NullBuildConfig)
    @test isdefined(Mimosa, :NullBuildResult)
    @test isdefined(Mimosa, :NullStrategy)
    @test isdefined(Mimosa, :MotifNullStrategy)
    @test isdefined(Mimosa, :ProfileNullStrategy)

    # Storage format versions
    @test isdefined(Mimosa, :MODEL_FORMAT_VERSION)
    @test isdefined(Mimosa, :NULL_FORMAT_VERSION)
end

# ---------------------------------------------------------------------------
# Model I/O: read models from files and write/read bundles
# ---------------------------------------------------------------------------
@testset "Downstream contract: model I/O" begin
    # Read PWM from MEME format (readmodel converts PFM→PWM internally)
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    @test pwm isa PWM

    # Read PFM file (returns PWM after conversion)
    pwm_from_pfm = readmodel(joinpath(EXAMPLES, "pif4.pfm"))
    @test pwm_from_pfm isa PWM

    # Read BaMM
    bamm = readmodel(joinpath(EXAMPLES, "foxa2.ihbcp"))
    @test bamm isa BaMM

    # Write and re-read a portable bundle (signature: writemodel(path, model))
    tmpdir = mktempdir()
    bundle_path = joinpath(tmpdir, "pwm_bundle")
    writemodel(bundle_path, pwm)
    @test isdir(bundle_path)
    @test isfile(joinpath(bundle_path, "manifest.toml"))

    loaded = readmodel(bundle_path)
    @test loaded isa PWM
    @test loaded.name == pwm.name
    @test size(loaded.weights) == size(pwm.weights)

    # Write and re-read BaMM bundle
    bamm_path = joinpath(tmpdir, "bamm_bundle")
    writemodel(bamm_path, bamm)
    loaded_bamm = readmodel(bamm_path)
    @test loaded_bamm isa BaMM
    @test loaded_bamm.order == bamm.order
    @test loaded_bamm.motif_length == bamm.motif_length
end

# ---------------------------------------------------------------------------
# Sequence reading
# ---------------------------------------------------------------------------
@testset "Downstream contract: sequence reading" begin
    batch, names = readsequences(joinpath(EXAMPLES, "foreground.fa"))
    @test batch isa EncodedSequenceBatch
    @test nsequences(batch) > 0
    @test length(names) == nsequences(batch)
    @test all(seqlength(batch, i) > 0 for i in 1:nsequences(batch))
end

# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
@testset "Downstream contract: scanning" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    batch, _names = readsequences(joinpath(EXAMPLES, "foreground.fa"))

    # Score bounds
    lo, hi = scorebounds(pwm)
    @test lo isa Float32
    @test hi isa Float32
    @test lo <= hi

    # Scan with different strand policies
    scores_fwd = scan(pwm, batch; strands=ForwardOnly())
    @test scores_fwd isa RaggedArray

    scores_best = scan(pwm, batch; strands=BestStrand())
    @test scores_best isa RaggedArray

    # BothStrands returns a StrandPair{RaggedArray}
    scores_both = scan(pwm, batch; strands=BothStrands())
    @test scores_both isa StrandPair
    @test scores_both.forward isa RaggedArray
    @test scores_both.reverse isa RaggedArray

    # Threaded scan == serial scan
    scores_thr = scan(pwm, batch; strands=BestStrand(), execution=ThreadedExecution(2))
    @test scores_best == scores_thr

    # In-place scan on a single sequence
    seq = sequence(batch, 1)
    n_pos = npositions(seqlength(batch, 1), length(pwm))
    dest = Vector{Float32}(undef, n_pos)
    scan!(dest, pwm, seq; strands=ForwardOnly())
    @test length(dest) == n_pos
    @test all(isfinite, dest)
end

# ---------------------------------------------------------------------------
# Direct motif comparison
# ---------------------------------------------------------------------------
@testset "Downstream contract: motif comparison" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "foxa2.meme"))

    # Test all metrics
    last_result = nothing
    for metric in (:pcc, :ed, :cosine)
        result = compare(pwm1, pwm2; metric=metric)
        @test result isa ComparisonResult
        @test result.query == pwm1.name
        @test result.target == pwm2.name
        @test result.metric == string(metric)
        last_result = result
    end

    # Self-comparison should give a high score
    self_result = compare(pwm1, pwm1; metric=:pcc)
    @test self_result.score >= last_result.score

    # Serialization
    json_str = to_json(last_result)
    @test json_str isa String
    dict = to_dict(last_result)
    @test dict isa Dict
end

# ---------------------------------------------------------------------------
# Profile comparison: one-to-one and one-to-many
# ---------------------------------------------------------------------------
@testset "Downstream contract: profile comparison" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "pif4.meme"))

    batch = make_random_sequences(20, 100; seed=42)

    # Motif-derived profile comparison
    result = compare(pwm1, pwm2, batch; metric=:co, search_range=5, window_radius=5)
    @test result isa ComparisonResult
    @test result.n_sites >= 0

    # One-to-many via prepared profile
    sp1 = ScoreProfile("q", scan(pwm1, batch; strands=BestStrand()))
    sp2 = ScoreProfile("t1", scan(pwm2, batch; strands=BestStrand()))
    sp3 = ScoreProfile("t2", scan(pwm1, batch; strands=BestStrand()))

    prepared = prepare_profile(sp1)
    @test prepared isa PreparedProfile

    results = compare(prepared, [sp2, sp3]; metric=:co, search_range=3, window_radius=2)
    @test length(results) == 2
    @test all(r isa ComparisonResult for r in results)
end

# ---------------------------------------------------------------------------
# Site extraction and PFM reconstruction
# ---------------------------------------------------------------------------
@testset "Downstream contract: sites and PFM reconstruction" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    batch = make_random_sequences(30, 200; seed=42)

    sites = selectsites(pwm, batch, BestPerSequence(); strands=BothStrands())
    @test sites isa SiteCollection

    pfm = reconstruct_pfm(pwm, batch, BestPerSequence(); pseudocount=Float32(1e-4))
    @test pfm isa AbstractMatrix{Float32}
    @test size(pfm, 1) == 4  # A, C, G, T rows

    # Higher-order model site extraction
    bamm = readmodel(joinpath(EXAMPLES, "foxa2.ihbcp"))
    @test site_start_offset(bamm) == bamm.order

    ho_sites = selectsites(bamm, batch, BestPerSequence(); strands=BothStrands())
    @test ho_sites isa SiteCollection

    if length(ho_sites) > 0
        ho_pfm = reconstruct_pfm(bamm, batch, BestPerSequence(); pseudocount=0.1f0)
        @test ho_pfm isa AbstractMatrix{Float32}
        @test size(ho_pfm, 1) == 4
    end
end

# ---------------------------------------------------------------------------
# Null distribution: build, save, load
# ---------------------------------------------------------------------------
@testset "Downstream contract: null distributions" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "foxa2.meme"))

    relations_str = "motif\tgroup\n$(pwm1.name)\tA\n$(pwm2.name)\tB\n"
    rel_path = joinpath(mktempdir(), "groups.tsv")
    write(rel_path, relations_str)
    relations = parse_group_relations(rel_path)

    # Motif null strategy
    models = [pwm1, pwm2]
    result = build_null(models, relations; strategy="motif", metric=:pcc)
    @test result isa NullBuildResult
    dist = result.distribution
    @test dist isa NullDistribution
    @test dist.strategy == "motif"
    @test dist.metric == "pcc"
    @test dist.model_collection_fingerprint !== nothing
    @test dist.relation_fingerprint !== nothing
    @test dist.sequence_fingerprint == "none"
    @test dist.background_fingerprint == "none"

    # Save and reload null (signature: savenull(path, dist))
    tmpdir = mktempdir()
    null_path = joinpath(tmpdir, "null_dist")
    savenull(null_path, dist)
    @test isdir(null_path)
    @test isfile(joinpath(null_path, "manifest.toml"))

    loaded_dist = loadnull(null_path)
    @test loaded_dist isa NullDistribution
    @test loaded_dist.strategy == dist.strategy
    @test loaded_dist.metric == dist.metric
    @test loaded_dist.n_null == dist.n_null
    @test loaded_dist.model_collection_fingerprint == dist.model_collection_fingerprint
end

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
@testset "Downstream contract: annotation" begin
    # Create multiple PWM models for a meaningful null distribution
    # (GEV fit requires at least 3 scores)
    weights_a = Float32[
        0.5 -0.3 0.8 -0.2 0.1 0.6
        -0.2 0.7 -0.5 0.3 0.8 -0.1
        0.1 -0.4 0.2 0.6 -0.3 0.5
        0.3 0.1 -0.6 0.4 0.2 -0.5
        -0.2 -0.3 -0.5 -0.2 -0.3 -0.5
    ]
    weights_b = Float32[
        0.3 0.2 -0.1 0.5 -0.4
        0.1 0.6 0.3 -0.2 0.7
        -0.4 0.1 0.4 0.3 -0.5
        0.2 -0.3 0.1 -0.1 0.8
        -0.3 -0.4 -0.2 -0.5 -0.6
    ]
    weights_c = Float32[
        -0.1 0.4 0.3 -0.5 0.2
        0.5 -0.2 0.6 0.1 -0.3
        0.3 0.5 -0.4 -0.2 0.7
        -0.5 0.1 0.2 0.8 -0.4
        -0.4 -0.3 -0.5 -0.1 -0.6
    ]
    weights_d = Float32[
        0.4 -0.1 0.7 -0.3 0.5
        -0.3 0.2 0.1 0.6 -0.4
        0.2 -0.5 0.4 -0.1 0.3
        -0.1 0.8 -0.2 -0.4 0.1
        -0.5 -0.4 -0.3 -0.6 -0.2
    ]
    bg = (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25))
    models = [
        PWM("m1", weights_a, bg),
        PWM("m2", weights_b, bg),
        PWM("m3", weights_c, bg),
        PWM("m4", weights_d, bg),
    ]

    relations_str = "motif\tgroup\nm1\tA\nm2\tB\nm3\tA\nm4\tB\n"
    rel_path = joinpath(mktempdir(), "groups.tsv")
    write(rel_path, relations_str)
    relations = parse_group_relations(rel_path)

    null_result = build_null(models, relations; strategy="motif", metric=:pcc)
    dist = null_result.distribution
    @test dist.n_null >= 3  # GEV fit requires at least 3 scores

    # Create comparison results to annotate
    results = [
        compare(models[1], models[2]; metric=:pcc),
        compare(models[1], models[3]; metric=:pcc),
        compare(models[2], models[4]; metric=:pcc),
    ]

    annotated = annotate_results(results, dist)
    @test length(annotated) == 3
    @test all(a isa AnnotatedResult for a in annotated)
    @test annotated[1].p_value !== nothing
    @test annotated[1].adj_p_value !== nothing
    @test annotated[1].e_value !== nothing
    @test annotated[1].null_n == dist.n_null
    @test annotated[1].null_estimator == "genextreme"
    @test annotated[1].null_id !== nothing

    # With effective_number_of_targets
    annotated2 = annotate_results(results, dist; effective_number_of_targets=50)
    @test annotated2[1].e_value ≈ annotated[1].p_value * 50

    # Serialization
    json_str = to_json(annotated[1])
    @test json_str isa String
    dict = to_dict(annotated[1])
    @test dict isa Dict
end

# ---------------------------------------------------------------------------
# Serial vs threaded equivalence
# ---------------------------------------------------------------------------
@testset "Downstream contract: serial == threaded" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    batch = make_random_sequences(20, 200; seed=123)

    serial = scan(pwm, batch; strands=BestStrand(), execution=SerialExecution())
    threaded = scan(pwm, batch; strands=BestStrand(), execution=ThreadedExecution(4))
    @test serial == threaded
end

# ---------------------------------------------------------------------------
# No internal access needed
# ---------------------------------------------------------------------------
@testset "Downstream contract: no internal access needed" begin
    # The fact that all above tests pass with only `using Mimosa`
    # (not `using Mimosa.Scanning` etc.) is the contract.
    @test true
end
