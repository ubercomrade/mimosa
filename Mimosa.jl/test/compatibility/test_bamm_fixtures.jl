using Test
using Mimosa
using JSON3

# REPO_ROOT, EXAMPLES, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

const BAMM_FIXTURES = joinpath(@__DIR__, "..", "fixtures")

"""
    _expected_bamm_2d(path)

Load the N-dimensional BaMM representation from the oracle fixture and
reshape it to 2D matching Python's C-order row-major layout.
The first `order+1` dimensions are reversed before reshape to match C-order.
"""
function _expected_bamm_2d(fixture_id::AbstractString, order::Int)
    expected = read_npy(joinpath(FIXTURE_COMPAT, "$(fixture_id)__representation.npy"))
    dims = length(size(expected))
    # Reverse context dimensions to match C-order reshape
    perm = vcat(reverse(1:(dims - 1)), dims)
    return reshape(permutedims(expected, Tuple(perm)), 5^(order + 1), size(expected)[end])
end

@testset "bamm_parse_myog" begin
    for order in [0, 1, 2]
        fixture_id = "bamm_parse_myog_order$(order)"
        model = read_bamm(joinpath(BAMM_FIXTURES, "myog.ihbcp"); order=order)
        expected = _expected_bamm_2d(fixture_id, order)
        md = fixture_metadata(fixture_id)

        @test model.name == md["name"]
        @test model.order == md["target_order"]
        @test model.motif_length == md["motif_length"]
        @test size(model.representation) == (5^(order + 1), md["motif_length"])
        @test model.representation ≈ expected
    end
end

@testset "bamm_parse_gata2" begin
    for order in [0, 1, 2]
        fixture_id = "bamm_parse_gata2_order$(order)"
        model = read_bamm(joinpath(BAMM_FIXTURES, "gata2.ihbcp"); order=order)
        expected = _expected_bamm_2d(fixture_id, order)
        md = fixture_metadata(fixture_id)

        @test model.name == md["name"]
        @test model.order == md["target_order"]
        @test model.motif_length == md["motif_length"]
        @test model.representation ≈ expected
    end
end

@testset "bamm_parse_foxa2" begin
    for order in [0, 1, 2]
        fixture_id = "bamm_parse_foxa2_order$(order)"
        model = read_bamm(joinpath(BAMM_FIXTURES, "foxa2.ihbcp"); order=order)
        expected = _expected_bamm_2d(fixture_id, order)
        md = fixture_metadata(fixture_id)

        @test model.name == md["name"]
        @test model.order == md["target_order"]
        @test model.motif_length == md["motif_length"]
        @test model.representation ≈ expected
    end
end

@testset "bamm_score_bounds" begin
    for (name, order) in [("myog", 1), ("myog", 0), ("gata2", 2)]
        fixture_id = "bamm_score_bounds_$(name)_order$(order)"
        model = read_bamm(joinpath(BAMM_FIXTURES, "$(name).ihbcp"); order=order)
        mn, mx = scorebounds(model)
        md = fixture_metadata(fixture_id)
        @test mn ≈ Float32(md["min_score"])
        @test mx ≈ Float32(md["max_score"])
    end
end

@testset "bamm_scan_fixtures" begin
    # Load input sequences
    input_values = read_npy(joinpath(FIXTURE_COMPAT, "bamm_scan_input_seed42__values.npy"))
    input_lengths = Int.(
        read_npy(joinpath(FIXTURE_COMPAT, "bamm_scan_input_seed42__lengths.npy"))
    )
    batch = from_padded(UInt8.(input_values), input_lengths)

    for (name, order) in [("myog", 1), ("myog", 0)]
        model = read_bamm(joinpath(BAMM_FIXTURES, "$(name).ihbcp"); order=order)

        # Forward scan
        fwd_id = "bamm_scan_forward_$(name)_order$(order)_seed42"
        fwd_scores = scan(model, batch; strands=ForwardOnly())
        fwd_expected = read_npy(joinpath(FIXTURE_COMPAT, "$(fwd_id)__values.npy"))
        fwd_md = fixture_metadata(fwd_id)

        @test fwd_md["motif_length"] == model.motif_length
        @test fwd_md["order"] == order

        for i in 1:nsequences(batch)
            n_pos = npositions(model, seqlength(batch, i))
            if n_pos > 0
                julia_row = row(fwd_scores, i)
                expected_row = fwd_expected[i, 1:n_pos]
                @test julia_row ≈ expected_row
            end
        end

        # Reverse scan
        rev_id = "bamm_scan_reverse_$(name)_order$(order)_seed42"
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

@testset "bamm_readmodel_auto" begin
    # Test that readmodel auto-detects .ihbcp format
    model = readmodel(joinpath(BAMM_FIXTURES, "myog.ihbcp"))
    @test model isa BaMM
    @test model.name == "myog"

    # With explicit order
    model0 = readmodel(joinpath(BAMM_FIXTURES, "myog.ihbcp"); order=0)
    @test model0 isa BaMM
    @test model0.order == 0
end
