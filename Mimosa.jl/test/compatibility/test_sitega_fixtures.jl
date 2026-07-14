using Test
using Mimosa

# REPO_ROOT, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

const SITEGA_FIXTURES = joinpath(@__DIR__, "..", "fixtures")

@testset "sitega_parse" begin
    for base in ["sitega", "sitega_gata2", "sitega_stat6"]
        fixture_id = "sitega_parse_$(base)"
        path = joinpath(SITEGA_FIXTURES, "$(base).mat")
        model = read_sitega(path)
        expected = read_npy(joinpath(FIXTURE_COMPAT, "$(fixture_id)__representation.npy"))
        md = fixture_metadata(fixture_id)

        @test model.name == md["name"]
        @test model.motif_length == md["motif_length"]
        @test size(model.representation) == (25, md["motif_length"])
        @test model.representation ≈ expected
    end
end

@testset "sitega_score_bounds" begin
    for base in ["sitega", "sitega_gata2", "sitega_stat6"]
        fixture_id = "sitega_score_bounds_$(base)"
        path = joinpath(SITEGA_FIXTURES, "$(base).mat")
        model = read_sitega(path)
        mn, mx = scorebounds(model)
        md = fixture_metadata(fixture_id)
        @test mn ≈ Float32(md["min_score"])
        @test mx ≈ Float32(md["max_score"])
    end
end

@testset "sitega_scan_fixtures" begin
    # Load input sequences
    input_values = read_npy(
        joinpath(FIXTURE_COMPAT, "sitega_scan_input_seed42__values.npy")
    )
    input_lengths = Int.(
        read_npy(joinpath(FIXTURE_COMPAT, "sitega_scan_input_seed42__lengths.npy"))
    )
    batch = from_padded(UInt8.(input_values), input_lengths)

    for base in ["sitega", "sitega_gata2"]
        model = read_sitega(joinpath(SITEGA_FIXTURES, "$(base).mat"))

        # Forward scan
        fwd_id = "sitega_scan_forward_$(base)_seed42"
        fwd_scores = scan(model, batch; strands=ForwardOnly())
        fwd_expected = read_npy(joinpath(FIXTURE_COMPAT, "$(fwd_id)__values.npy"))
        fwd_md = fixture_metadata(fwd_id)

        @test fwd_md["motif_length"] == model.motif_length
        @test fwd_md["kmer"] == 2

        for i in 1:nsequences(batch)
            n_pos = npositions(model, seqlength(batch, i))
            if n_pos > 0
                julia_row = row(fwd_scores, i)
                expected_row = fwd_expected[i, 1:n_pos]
                @test julia_row ≈ expected_row
            end
        end

        # Reverse scan
        rev_id = "sitega_scan_reverse_$(base)_seed42"
        rev_scores = scan(model, batch; strands=ReverseOnly())
        rev_expected = read_npy(joinpath(FIXTURE_COMPAT, "$(rev_id)__values.npy"))

        for i in 1:nsequences(batch)
            n_pos = npositions(model, seqlength(batch, i))
            if n_pos > 0
                julia_row = row(rev_scores, i)
                expected_row = rev_expected[i, 1:n_pos]
                @test julia_row ≈ expected_row
            end
        end
    end
end

@testset "sitega_readmodel_auto" begin
    # Test that readmodel auto-detects .mat format
    model = readmodel(joinpath(SITEGA_FIXTURES, "sitega_gata2.mat"))
    @test model isa SiteGA
    @test model.name == "GATA2"
end
