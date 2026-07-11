using Test
using Mimosa
using JSON3

# REPO_ROOT, EXAMPLES, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

# Convert NPY Int8 array to UInt8 for sequence comparison
# (defined in test_scan_fixtures.jl, reused here)
# _to_uint8_matrix is imported from the scan fixtures scope

@testset "sites_input_seed42" begin
    expected_values = read_npy(joinpath(FIXTURE_COMPAT, "sites_input_seed42__values.npy"))
    expected_lengths = read_npy(joinpath(FIXTURE_COMPAT, "sites_input_seed42__lengths.npy"))
    md = fixture_metadata("sites_input_seed42")

    expected_u8 = _to_uint8_matrix(expected_values)
    global SITES_INPUT_BATCH = from_padded(expected_u8, expected_lengths)
    @test nsequences(SITES_INPUT_BATCH) == md["n_sequences"]
    for i in 1:nsequences(SITES_INPUT_BATCH)
        @test seqlength(SITES_INPUT_BATCH, i) == expected_lengths[i]
    end
end

@testset "sites_best_pif4_seed42" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    md = fixture_metadata("sites_best_pif4_seed42")

    # Extract best-per-sequence sites with BothStrands
    coll = selectsites(pwm, SITES_INPUT_BATCH, BestPerSequence(); strands=BothStrands())

    @test length(coll) == md["n_sites"]

    # Load oracle arrays
    expected_seq_index = read_npy(
        joinpath(FIXTURE_COMPAT, "sites_best_pif4_seed42__seq_index.npy")
    )
    expected_start = read_npy(joinpath(FIXTURE_COMPAT, "sites_best_pif4_seed42__start.npy"))
    expected_end = read_npy(joinpath(FIXTURE_COMPAT, "sites_best_pif4_seed42__end.npy"))
    expected_score = read_npy(joinpath(FIXTURE_COMPAT, "sites_best_pif4_seed42__score.npy"))
    expected_log_tail = read_npy(
        joinpath(FIXTURE_COMPAT, "sites_best_pif4_seed42__log_tail.npy")
    )

    motif_length = md["motif_length"]

    # Compare: Julia uses 1-based, oracle uses 0-based
    for h in 1:length(coll)
        # seq_index: Julia 1-based → oracle 0-based
        @test coll.seq_indices[h] == expected_seq_index[h] + 1
        # start: Julia 1-based → oracle 0-based
        @test coll.starts[h] == expected_start[h] + 1
        # end = start + motif_length (Julia: starts[h] + motif_length - 1 inclusive)
        @test coll.starts[h] + motif_length - 1 == expected_end[h]
        # score
        @test coll.scores[h] ≈ expected_score[h]
    end
end

@testset "pfm_reconstruction_best_pif4_seed42" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    md = fixture_metadata("pfm_reconstruction_best_pif4_seed42")

    # Reconstruct PFM from best sites with pseudocount=0.25
    pfm = reconstruct_pfm(
        pwm,
        SITES_INPUT_BATCH,
        BestPerSequence();
        pseudocount=Float32(md["pseudocount"]),
        strands=BothStrands(),
    )

    expected_pfm = read_npy(
        joinpath(FIXTURE_COMPAT, "pfm_reconstruction_best_pif4_seed42__pfm.npy")
    )

    @test size(pfm) == tuple(md["shape"]...)
    @test pfm ≈ expected_pfm
end

@testset "sites determinism" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))

    coll1 = selectsites(pwm, SITES_INPUT_BATCH, BestPerSequence(); strands=BothStrands())
    coll2 = selectsites(pwm, SITES_INPUT_BATCH, BestPerSequence(); strands=BothStrands())

    @test coll1 == coll2
end

@testset "sites non-mutation" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))

    batch_copy = EncodedSequenceBatch(
        copy(SITES_INPUT_BATCH.data), copy(SITES_INPUT_BATCH.offsets)
    )

    selectsites(pwm, SITES_INPUT_BATCH, BestPerSequence(); strands=BothStrands())

    @test SITES_INPUT_BATCH.data == batch_copy.data
    @test SITES_INPUT_BATCH.offsets == batch_copy.offsets
end
