using Test
using Mimosa
using JSON3

# REPO_ROOT, EXAMPLES, FIXTURE_COMPAT, read_npy, fixture_metadata
# are defined in runtests.jl

# Convert NPY Int8 array to UInt8 for sequence comparison
_to_uint8(arr) = reinterpret(UInt8, convert(Array{Int8}, arr))
_to_uint8_matrix(arr) = reinterpret(UInt8, convert(Array{Int8,ndims(arr)}, arr))

@testset "fasta_read_foreground" begin
    batch, names = read_fasta(joinpath(EXAMPLES, "foreground.fa"))
    md = fixture_metadata("fasta_read_foreground")

    # Compare encoded values and lengths
    expected_values = read_npy(
        joinpath(FIXTURE_COMPAT, "fasta_read_foreground__values.npy")
    )
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "fasta_read_foreground__lengths.npy")
    )

    # NPY reader returns Int8; convert to UInt8 for comparison
    expected_u8 = _to_uint8_matrix(expected_values)

    # Compare each sequence's encoded bytes
    for i in 1:nsequences(batch)
        @test seqlength(batch, i) == expected_lengths[i]
        for j in 1:seqlength(batch, i)
            @test sequence(batch, i)[j] == expected_u8[i, j]
        end
    end
end

@testset "random_sequence_batch_seed127" begin
    expected_values = read_npy(
        joinpath(FIXTURE_COMPAT, "random_sequence_batch_seed127__values.npy")
    )
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "random_sequence_batch_seed127__lengths.npy")
    )

    # Build batch from padded oracle data (convert Int8 → UInt8)
    expected_u8 = _to_uint8_matrix(expected_values)
    batch = from_padded(expected_u8, expected_lengths)
    md = fixture_metadata("random_sequence_batch_seed127")
    @test nsequences(batch) == md["n_sequences"]
    for i in 1:nsequences(batch)
        @test seqlength(batch, i) == expected_lengths[i]
    end

    # Round-trip: to_padded should reproduce the oracle
    padded, lengths = to_padded(batch)
    @test lengths == collect(expected_lengths)
    @test padded == expected_u8
end

@testset "pwm_scan_input_seed42" begin
    expected_values = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_input_seed42__values.npy")
    )
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_input_seed42__lengths.npy")
    )
    md = fixture_metadata("pwm_scan_input_seed42")

    # Build batch from oracle data (convert Int8 → UInt8)
    expected_u8 = _to_uint8_matrix(expected_values)
    global SCAN_INPUT_BATCH = from_padded(expected_u8, expected_lengths)
    @test nsequences(SCAN_INPUT_BATCH) == md["n_sequences"]
    for i in 1:nsequences(SCAN_INPUT_BATCH)
        @test seqlength(SCAN_INPUT_BATCH, i) == expected_lengths[i]
    end
end

@testset "pwm_scan_forward_pif4_seed42" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    expected_values = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_forward_pif4_seed42__values.npy")
    )
    expected_mask = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_forward_pif4_seed42__mask.npy")
    )
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_forward_pif4_seed42__lengths.npy")
    )

    # Scan the input batch
    result = scan(pwm, SCAN_INPUT_BATCH; strands=ForwardOnly())

    @test nrows(result) == length(expected_lengths)

    for i in 1:nrows(result)
        len = Int(expected_lengths[i])
        @test rowlength(result, i) == len
        if len > 0
            expected_row = collect(expected_values[i, 1:len])
            @test row(result, i) ≈ expected_row
        end
    end
end

@testset "pwm_scan_reverse_pif4_seed42" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    expected_values = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_reverse_pif4_seed42__values.npy")
    )
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_reverse_pif4_seed42__lengths.npy")
    )

    result = scan(pwm, SCAN_INPUT_BATCH; strands=ReverseOnly())

    @test nrows(result) == length(expected_lengths)

    for i in 1:nrows(result)
        len = Int(expected_lengths[i])
        @test rowlength(result, i) == len
        if len > 0
            expected_row = collect(expected_values[i, 1:len])
            @test row(result, i) ≈ expected_row
        end
    end
end

@testset "pwm_scan_both_pif4_seed42" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    expected_values = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_both_pif4_seed42__values.npy")
    )
    expected_lengths = read_npy(
        joinpath(FIXTURE_COMPAT, "pwm_scan_both_pif4_seed42__lengths.npy")
    )

    md = fixture_metadata("pwm_scan_both_pif4_seed42")
    @test size(expected_values) == tuple(md["bundle_shape"]...)

    result = scan(pwm, SCAN_INPUT_BATCH; strands=BothStrands())

    @test nrows(result.forward) == length(expected_lengths)
    @test nrows(result.reverse) == length(expected_lengths)

    for i in 1:nrows(result.forward)
        len = Int(expected_lengths[i])
        @test rowlength(result.forward, i) == len
        @test rowlength(result.reverse, i) == len
        if len > 0
            # Oracle values shape: (2, n_seqs, n_positions)
            # values[1, i, :] = forward, values[2, i, :] = reverse
            expected_fwd = collect(expected_values[1, i, 1:len])
            expected_rev = collect(expected_values[2, i, 1:len])
            @test row(result.forward, i) ≈ expected_fwd
            @test row(result.reverse, i) ≈ expected_rev
        end
    end
end

@testset "pwm_scan_best equals max of both strands" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))

    fwd = scan(pwm, SCAN_INPUT_BATCH; strands=ForwardOnly())
    rev = scan(pwm, SCAN_INPUT_BATCH; strands=ReverseOnly())
    best = scan(pwm, SCAN_INPUT_BATCH; strands=BestStrand())

    for i in 1:nrows(fwd)
        if rowlength(fwd, i) > 0
            @test row(best, i) == max.(row(fwd, i), row(rev, i))
        end
    end
end

@testset "pwm_scan forward equals scan! forward" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))

    for i in 1:nsequences(SCAN_INPUT_BATCH)
        seq = sequence(SCAN_INPUT_BATCH, i)
        alloc = scan(pwm, seq; strands=ForwardOnly())
        dest = Vector{Float32}(undef, length(alloc))
        scan!(dest, pwm, seq; strands=ForwardOnly())
        @test dest == alloc
    end
end
