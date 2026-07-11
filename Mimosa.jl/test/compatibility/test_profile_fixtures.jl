using Test
using Mimosa
using JSON3

# REPO_ROOT, EXAMPLES, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

@testset "normalization_log_tail_pif4_seed42" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))

    # Build the same random batch as the Python oracle (50 seqs × 200bp, seed=42)
    # The oracle used make_random_sequence_batch(50, 200, seed=42)
    # We read the oracle's scan input fixture for the sequences
    expected_seq_values = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_input_seed42__values.npy")
    )
    expected_seq_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_input_seed42__lengths.npy")
    )

    # Convert Int8 → UInt8 for sequence comparison
    expected_u8 = reinterpret(UInt8, convert(Array{Int8}, expected_seq_values))
    # The oracle used 50 sequences of length 200 for normalization
    # But the scan input fixture has different parameters — let's check
    # Actually, the normalization fixture uses make_random_sequence_batch(50, 200, seed=42)
    # The scan input fixture uses different params. Let me just use the oracle's flat scores directly.

    # Read the oracle flat scores and table
    expected_flat = read_npy(
        joinpath(FIXTURE_COMPAT, "normalization_log_tail_pif4_seed42__flat_scores.npy")
    )
    expected_table = read_npy(
        joinpath(FIXTURE_COMPAT, "normalization_log_tail_pif4_seed42__table.npy")
    )

    md = fixture_metadata("normalization_log_tail_pif4_seed42")
    @test length(expected_flat) == md["n_scores"]
    @test size(expected_table) == tuple(md["table_shape"]...)

    # Fit the table from the oracle's flat scores
    table = fit(EmpiricalLogTail(), vec(expected_flat))

    # Compare the table
    @test length(table.scores) == size(expected_table, 1)
    @test length(table.log_tail) == size(expected_table, 1)

    for i in 1:length(table.scores)
        @test table.scores[i] ≈ expected_table[i, 1] atol = 1e-6
        @test table.log_tail[i] ≈ expected_table[i, 2] atol = 1e-5
    end
end

@testset "score_profile_read" begin
    # Test reading score profiles
    sp1 = read_scores(joinpath(EXAMPLES, "scores_1.fasta"))
    @test sp1.name == "scores_1"

    expected_values = read_npy(joinpath(FIXTURE_COMPAT, "score_profile_read_1__values.npy"))
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "score_profile_read_1__lengths.npy")
    )
    expected_mask = read_npy(joinpath(FIXTURE_COMPAT, "score_profile_read_1__mask.npy"))

    md = fixture_metadata("score_profile_read_1")
    @test nrows(sp1.scores) == md["n_profiles"]

    # Compare each row
    for i in 1:nrows(sp1.scores)
        len = Int(expected_lengths[i])
        @test rowlength(sp1.scores, i) == len
        if len > 0
            expected_row = collect(expected_values[i, 1:len])
            @test row(sp1.scores, i) ≈ expected_row
        end
    end

    # Also test scores_2
    sp2 = read_scores(joinpath(EXAMPLES, "scores_2.fasta"))
    @test sp2.name == "scores_2"
    @test nrows(sp2.scores) == md["n_profiles"]
end

@testset "profile_comparison_scores_zero_shift" begin
    sp1 = read_scores(joinpath(EXAMPLES, "scores_1.fasta"))
    sp2 = read_scores(joinpath(EXAMPLES, "scores_2.fasta"))

    for metric_name_str in ("co", "co_rowwise", "dice", "dice_rowwise", "cosine")
        fixture_id = "profile_comparison_scores_$(metric_name_str)_zero_shift"
        md = fixture_metadata(fixture_id)

        result = compare(
            sp1, sp2; metric=Symbol(metric_name_str), search_range=0, window_radius=0
        )

        @test result.query == md["query"]
        @test result.target == md["target"]
        @test result.metric == md["metric"]
        @test result.orientation == md["orientation"]
        @test result.offset == md["offset"]
        @test result.n_sites == md["n_sites"]
        @test Float64(result.score) ≈ md["score"] atol = 1e-5
    end
end
