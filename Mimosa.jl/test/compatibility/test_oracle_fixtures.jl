using Test
using Mimosa
using JSON3

const REPO_ROOT = joinpath(dirname(dirname(@__DIR__)), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")
const FIXTURE_COMPAT = joinpath(REPO_ROOT, "tests", "fixtures", "compatibility")

include(joinpath(@__DIR__, "..", "npy_reader.jl"))
using .NPYReader: read_npy

const MANIFEST = JSON3.read(read(joinpath(FIXTURE_COMPAT, "manifest.json"), String))

function fixture_metadata(id::AbstractString)
    for f in MANIFEST["fixtures"]
        f["id"] == id && return f["metadata"]
    end
    error("fixture $id not found in manifest.")
end

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
    @test pwm4 == expected
end

@testset "pwm_reverse_complement_pif4" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"); index=0)
    pwm4 = pfm_to_pwm(pfm.frequencies)
    forward_expected = read_npy(joinpath(FIXTURE_COMPAT, "pwm_reverse_complement_pif4__forward.npy"))
    @test pwm4 == forward_expected
    rc = reverse_complement(pwm4)
    reverse_expected = read_npy(joinpath(FIXTURE_COMPAT, "pwm_reverse_complement_pif4__reverse.npy"))
    @test rc == reverse_expected
end

@testset "pwm_score_bounds_pif4" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    mn, mx = scorebounds(pwm)
    md = fixture_metadata("pwm_score_bounds_pif4")
    @test mn ≈ Float32(md["min_score"])
    @test mx ≈ Float32(md["max_score"])
end

@testset "motif_alignment_self_pif4_pcc" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="pcc")
    md = fixture_metadata("motif_alignment_self_pif4_pcc")
    @test result.metric == md["metric"]
    @test result.offset == md["offset"]
    @test result.orientation == md["orientation"]
    @test result.query == md["query"]
    @test result.target == md["target"]
    @test result.score ≈ Float32(md["score"])
end

@testset "motif_alignment_self_pif4_ed" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="ed")
    md = fixture_metadata("motif_alignment_self_pif4_ed")
    @test result.offset == md["offset"]
    @test result.orientation == md["orientation"]
    @test result.score ≈ Float32(md["score"])
end

@testset "motif_alignment_self_pif4_cosine" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="cosine")
    md = fixture_metadata("motif_alignment_self_pif4_cosine")
    @test result.offset == md["offset"]
    @test result.orientation == md["orientation"]
    @test result.score ≈ Float32(md["score"])
end

@testset "motif_alignment_pif4_vs_gata2_pcc" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "gata2.meme"))
    result = compare(pwm1, pwm2; metric="pcc")
    md = fixture_metadata("motif_alignment_pif4_vs_gata2_pcc")
    @test result.offset == md["offset"]
    @test result.orientation == md["orientation"]
    @test result.score ≈ Float32(md["score"])
end

@testset "motif_alignment_pif4_vs_gata2_ed" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "gata2.meme"))
    result = compare(pwm1, pwm2; metric="ed")
    md = fixture_metadata("motif_alignment_pif4_vs_gata2_ed")
    @test result.offset == md["offset"]
    @test result.orientation == md["orientation"]
    @test result.score ≈ Float32(md["score"])
end

@testset "motif_alignment_pif4_vs_gata2_cosine" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "gata2.meme"))
    result = compare(pwm1, pwm2; metric="cosine")
    md = fixture_metadata("motif_alignment_pif4_vs_gata2_cosine")
    @test result.offset == md["offset"]
    @test result.orientation == md["orientation"]
    @test result.score ≈ Float32(md["score"])
end

@testset "cli_motif_self_pif4_pcc JSON" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    result = compare(pwm, pwm; metric="pcc")
    json_str = Mimosa.to_json(result)
    expected = JSON3.read(json_str)
    md = fixture_metadata("cli_motif_self_pif4_pcc")
    @test Set(keys(expected)) == Set(md["keys"])
    @test expected["score"] == md["score"] || expected["score"] ≈ md["score"]
    @test expected["offset"] == md["offset"]
    @test expected["orientation"] == md["orientation"]
end