using Test
using Mimosa

# REPO_ROOT, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

const DIMONT_FIXTURES = joinpath(@__DIR__, "..", "fixtures")

@testset "dimont_parse" begin
    for base in [
        "exampleD-model-1",
        "PEAKS036274_FOXA1_P35582_MACS2-model-1",
        "PEAKS038885_CEBPB_P28033_MACS2-model-1",
        "PEAKS038885_CEBPB_P28033_MACS2-model-2",
        "stat_dimont-model-1",
    ]
        fixture_id = "dimont_parse_$(base)"
        path = joinpath(DIMONT_FIXTURES, "$(base).xml")
        model = read_dimont(path)
        expected = read_npy(joinpath(FIXTURE_COMPAT, "$(fixture_id)__representation.npy"))
        md = fixture_metadata(fixture_id)

        @test model.name == md["name"]
        @test model.motif_length == md["motif_length"]
        @test model.span == md["span"]
        @test size(model.representation) == (size(expected, 1), md["motif_length"])
        @test model.representation ≈ expected
    end
end

@testset "dimont_score_bounds" begin
    for base in [
        "exampleD-model-1",
        "PEAKS036274_FOXA1_P35582_MACS2-model-1",
        "PEAKS038885_CEBPB_P28033_MACS2-model-1",
        "PEAKS038885_CEBPB_P28033_MACS2-model-2",
        "stat_dimont-model-1",
    ]
        fixture_id = "dimont_score_bounds_$(base)"
        path = joinpath(DIMONT_FIXTURES, "$(base).xml")
        model = read_dimont(path)
        mn, mx = scorebounds(model)
        md = fixture_metadata(fixture_id)
        @test mn ≈ Float32(md["min_score"])
        @test mx ≈ Float32(md["max_score"])
    end
end

@testset "dimont_scan_fixtures" begin
    # Load input sequences
    input_values = read_npy(
        joinpath(FIXTURE_COMPAT, "dimont_scan_input_seed42__values.npy")
    )
    input_lengths = Int.(
        read_npy(joinpath(FIXTURE_COMPAT, "dimont_scan_input_seed42__lengths.npy"))
    )
    batch = from_padded(UInt8.(input_values), input_lengths)

    for base in ["exampleD-model-1", "stat_dimont-model-1"]
        model = read_dimont(joinpath(DIMONT_FIXTURES, "$(base).xml"))

        # Forward scan
        fwd_id = "dimont_scan_forward_$(base)_seed42"
        fwd_scores = scan(model, batch; strands=ForwardOnly())
        fwd_expected = read_npy(joinpath(FIXTURE_COMPAT, "$(fwd_id)__values.npy"))
        fwd_md = fixture_metadata(fwd_id)

        @test fwd_md["motif_length"] == model.motif_length
        @test fwd_md["span"] == model.span
        @test fwd_md["kmer"] == model.span + 1

        for i in 1:nsequences(batch)
            n_pos = npositions(model, seqlength(batch, i))
            if n_pos > 0
                julia_row = row(fwd_scores, i)
                expected_row = fwd_expected[i, 1:n_pos]
                @test julia_row ≈ expected_row
            end
        end

        # Reverse scan
        rev_id = "dimont_scan_reverse_$(base)_seed42"
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

@testset "dimont_readmodel_auto" begin
    # Test that readmodel auto-detects .xml format
    model = readmodel(joinpath(DIMONT_FIXTURES, "stat_dimont-model-1.xml"))
    @test model isa Dimont
    @test model.name == "stat_dimont-model-1"
end
