# Tests for B2 (encoded sequence validation before unsafe kernels)
# and B3 (model constructor invariant enforcement).

using Test
using Mimosa
using Random

const REPO_ROOT = joinpath(dirname(dirname(@__DIR__)), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

# ── B2: EncodedSequenceBatch code validation ──────────────────────────────

@testset "B2: EncodedSequenceBatch rejects invalid codes" begin
    # Valid codes (0..4) should be accepted.
    @test_nowarn EncodedSequenceBatch(UInt8[0, 1, 2, 3, 4], Int[1, 6])
    @test_nowarn EncodedSequenceBatch(UInt8[0, 0, 0], Int[1, 4])

    # Invalid code 0x05 should be rejected.
    @test_throws InvariantError EncodedSequenceBatch(UInt8[0, 1, 5], Int[1, 4])
    # Invalid code 0xFF should be rejected.
    @test_throws InvariantError EncodedSequenceBatch(UInt8[0, 255, 2], Int[1, 4])
    # Invalid code in second sequence.
    @test_throws InvariantError EncodedSequenceBatch(
        UInt8[0, 1, 2, 3, 0, 6, 2], Int[1, 4, 8]
    )

    # Rows constructor also validates.
    @test_throws InvariantError EncodedSequenceBatch([UInt8[0, 1, 2], UInt8[3, 7, 0]])
    @test_throws InvariantError EncodedSequenceBatch([UInt8[0, 1, 2], UInt8[3, 0xff, 0]])
end

@testset "B2: from_padded rejects invalid codes" begin
    # Valid padded matrix.
    valid = UInt8[0 1 2 3; 3 2 1 0]
    @test_nowarn from_padded(valid, Int[4, 4])

    # Invalid code in padded matrix.
    invalid = UInt8[0 1 2 3; 3 2 1 5]
    @test_throws InvariantError from_padded(invalid, Int[4, 4])

    # Invalid padding value.
    @test_throws InvariantError from_padded(UInt8[0 1; 2 3], Int[2, 2]; padding=0x05)

    # Negative length.
    @test_throws ArgumentError from_padded(UInt8[0 1; 2 3], Int[-1, 2])

    # Length exceeds matrix width.
    @test_throws ArgumentError from_padded(UInt8[0 1; 2 3], Int[5, 2])
end

@testset "B2: reverse_complement! aliasing detection" begin
    # Normal (non-aliasing) use works.
    src = UInt8[0, 1, 2, 3]  # ACGT
    dest = similar(src)
    reverse_complement!(dest, src)
    @test dest == UInt8[0, 1, 2, 3]  # ACGT is RC palindrome
    @test src == UInt8[0, 1, 2, 3]   # src unchanged

    # Aliasing: dest === src should throw.
    aliased = UInt8[0, 1, 2, 3]
    @test_throws ArgumentError reverse_complement!(aliased, aliased)

    # dest larger than src is fine.
    src2 = UInt8[0, 0, 0, 0]
    dest2 = Vector{UInt8}(undef, 6)
    reverse_complement!(dest2, src2)
    @test dest2[1:4] == UInt8[3, 3, 3, 3]
end

@testset "B2: scan kernel destination validation" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    W = length(pwm)
    n_pos = npositions(length(seq), W)

    # Correct destination size works.
    dest = Vector{Float32}(undef, n_pos)
    @test_nowarn scan_forward!(dest, pwm.weights, seq, n_pos)
    @test_nowarn scan_reverse!(dest, pwm.weights, seq, n_pos)
    @test_nowarn scan_best!(dest, pwm.weights, seq, n_pos)

    fwd = Vector{Float32}(undef, n_pos)
    rev = Vector{Float32}(undef, n_pos)
    @test_nowarn scan_both!(fwd, rev, pwm.weights, seq, n_pos)

    # Short destination should throw.
    short_dest = Vector{Float32}(undef, max(n_pos - 1, 0))
    if n_pos > 0
        @test_throws ArgumentError scan_forward!(short_dest, pwm.weights, seq, n_pos)
        @test_throws ArgumentError scan_reverse!(short_dest, pwm.weights, seq, n_pos)
        @test_throws ArgumentError scan_best!(short_dest, pwm.weights, seq, n_pos)
    end

    if n_pos > 0
        short_fwd = Vector{Float32}(undef, n_pos - 1)
        @test_throws ArgumentError scan_both!(short_fwd, rev, pwm.weights, seq, n_pos)
        @test_throws ArgumentError scan_both!(fwd, short_fwd, pwm.weights, seq, n_pos)
    end

    # Negative n_pos should throw.
    @test_throws ArgumentError scan_forward!(dest, pwm.weights, seq, -1)
    @test_throws ArgumentError scan_reverse!(dest, pwm.weights, seq, -1)
    @test_throws ArgumentError scan_best!(dest, pwm.weights, seq, -1)
    @test_throws ArgumentError scan_both!(fwd, rev, pwm.weights, seq, -1)
end

@testset "B2: higher-order scan kernel destination validation" begin
    # Create a simple BaMM order=0 (5 rows for order=0).
    rep = Matrix{Float32}(undef, 5, 3)
    rep .= 0.0
    model = BaMM("test", rep, 0, 3)
    seq = UInt8[0, 1, 2, 3, 0, 1, 2, 3]
    n_pos = npositions_bamm(length(seq), model)

    dest = Vector{Float32}(undef, n_pos)
    @test_nowarn Mimosa._ho_scan_forward!(dest, model.representation, 1, 0, 3, seq, n_pos)

    # Short dest.
    if n_pos > 0
        short = Vector{Float32}(undef, n_pos - 1)
        @test_throws ArgumentError Mimosa._ho_scan_forward!(
            short, model.representation, 1, 0, 3, seq, n_pos
        )
    end

    # Negative n_pos.
    @test_throws ArgumentError Mimosa._ho_scan_forward!(
        dest, model.representation, 1, 0, 3, seq, -1
    )
end

@testset "B2: empty and short sequences" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    W = length(pwm)

    # Empty sequence: 0 scan positions.
    empty_seq = UInt8[]
    @test scan(pwm, empty_seq) == Float32[]

    # Sequence shorter than motif: 0 scan positions.
    short_seq = UInt8[0, 1, 2]
    @test scan(pwm, short_seq) == Float32[]

    # Sequence exactly motif width: 1 scan position.
    exact_seq = UInt8[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    @test length(exact_seq) == W
    result = scan(pwm, exact_seq)
    @test length(result) == 1

    # Empty batch.
    batch = empty_sequence_batch()
    result = scan(pwm, batch)
    @test nrows(result) == 0

    # Batch with empty rows.
    batch2 = EncodedSequenceBatch([UInt8[], UInt8[0, 1, 2, 3]])
    result2 = scan(pwm, batch2)
    @test nrows(result2) == 2
    @test rowlength(result2, 1) == 0
end

@testset "B2: allocating/in-place scan equivalence" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")

    for strands in (ForwardOnly(), ReverseOnly(), BestStrand())
        alloc_result = scan(pwm, seq; strands=strands)
        dest = Vector{Float32}(undef, length(alloc_result))
        scan!(dest, pwm, seq; strands=strands)
        @test dest == alloc_result
    end
end

@testset "B2: fuzzed valid encoded inputs" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    rng = MersenneTwister(2024)

    for _ in 1:50
        len = rand(rng, 20:200)
        # Generate valid encoded sequences with random codes 0..4.
        seq = UInt8[rand(rng, 0:4) for _ in 1:len]
        # All codes are valid, so this should not throw.
        result = scan(pwm, seq)
        @test length(result) == npositions(len, length(pwm))
        @test all(isfinite, result)
    end
end

# ── B3: Model constructor invariant enforcement ───────────────────────────

@testset "B3: PFM constructor validation" begin
    # Valid PFM.
    valid = Float32[0.25 0.5; 0.25 0.25; 0.25 0.15; 0.25 0.1]
    @test_nowarn PFM("test", valid)

    # Wrong number of rows.
    @test_throws ModelDimensionError PFM("test", Float32[0.5 0.5; 0.5 0.5])

    # Zero width.
    @test_throws ModelDimensionError PFM("test", Matrix{Float32}(undef, 4, 0))

    # Non-finite values.
    @test_throws ModelFormatError PFM(
        "test", Float32[0.25 Inf; 0.25 0.25; 0.25 0.25; 0.25 0.25]
    )

    # Negative values.
    @test_throws ModelFormatError PFM(
        "test", Float32[-0.1 0.5; 0.25 0.25; 0.25 0.25; 0.25 0.25]
    )
end

@testset "B3: PWM constructor validation" begin
    # Valid PWM.
    weights = Matrix{Float32}(undef, 5, 4)
    weights .= 0.0
    bg = (0.25f0, 0.25f0, 0.25f0, 0.25f0)
    @test_nowarn PWM("test", weights, bg)

    # Wrong number of rows.
    @test_throws ModelDimensionError PWM("test", Matrix{Float32}(undef, 4, 4), bg)

    # Zero width.
    @test_throws ModelDimensionError PWM("test", Matrix{Float32}(undef, 5, 0), bg)

    # Non-finite weights.
    bad_weights = Matrix{Float32}(undef, 5, 4)
    bad_weights .= 0.0
    bad_weights[1, 1] = Inf
    @test_throws ModelFormatError PWM("test", bad_weights, bg)

    # Negative background.
    @test_throws ModelFormatError PWM("test", weights, (-0.1f0, 0.3f0, 0.4f0, 0.4f0))

    # Non-finite background.
    @test_throws ModelFormatError PWM("test", weights, (Inf32, 0.25f0, 0.25f0, 0.25f0))

    # Background sum != 1.
    @test_throws ModelFormatError PWM("test", weights, (0.1f0, 0.1f0, 0.1f0, 0.1f0))
end

@testset "B3: BaMM constructor validation" begin
    # Valid BaMM order=0: 5 rows.
    rep0 = Matrix{Float32}(undef, 5, 4)
    rep0 .= 0.0
    @test_nowarn BaMM("test", rep0, 0, 4)

    # Valid BaMM order=1: 25 rows.
    rep1 = Matrix{Float32}(undef, 25, 4)
    rep1 .= 0.0
    @test_nowarn BaMM("test", rep1, 1, 4)

    # Negative order.
    @test_throws ModelDimensionError BaMM("test", rep0, -1, 4)

    # Order too high (guard against blow-up).
    @test_throws ModelDimensionError BaMM("test", rep0, 11, 4)

    # Wrong row count for order.
    @test_throws ModelDimensionError BaMM("test", Matrix{Float32}(undef, 10, 4), 1, 4)

    # Non-finite values.
    rep_bad = copy(rep0)
    rep_bad[1, 1] = NaN
    @test_throws ModelFormatError BaMM("test", rep_bad, 0, 4)

    # Non-positive motif_length.
    @test_throws ModelDimensionError BaMM("test", Matrix{Float32}(undef, 5, 0), 0, 0)
end

@testset "B3: SiteGA constructor validation" begin
    # Valid SiteGA: 25 rows.
    rep = Matrix{Float32}(undef, 25, 6)
    rep .= 0.0
    @test_nowarn SiteGA("test", rep, 6)

    # Wrong row count.
    @test_throws ModelDimensionError SiteGA("test", Matrix{Float32}(undef, 20, 6), 6)

    # Non-positive motif_length.
    @test_throws ModelDimensionError SiteGA("test", Matrix{Float32}(undef, 25, 0), 0)

    # Non-finite values.
    rep_bad = copy(rep)
    rep_bad[1, 1] = Inf
    @test_throws ModelFormatError SiteGA("test", rep_bad, 6)
end

@testset "B3: Dimont constructor validation" begin
    # Valid Dimont span=0: 5 rows.
    rep0 = Matrix{Float32}(undef, 5, 4)
    rep0 .= 0.0
    @test_nowarn Dimont("test", rep0, 0, 4)

    # Valid Dimont span=1: 25 rows.
    rep1 = Matrix{Float32}(undef, 25, 4)
    rep1 .= 0.0
    @test_nowarn Dimont("test", rep1, 1, 4)

    # Negative span.
    @test_throws ModelDimensionError Dimont("test", rep0, -1, 4)

    # Span too high.
    @test_throws ModelDimensionError Dimont("test", rep0, 11, 4)

    # Wrong row count for span.
    @test_throws ModelDimensionError Dimont("test", Matrix{Float32}(undef, 10, 4), 1, 4)

    # Non-finite values.
    rep_bad = copy(rep0)
    rep_bad[1, 1] = NaN
    @test_throws ModelFormatError Dimont("test", rep_bad, 0, 4)
end

@testset "B3: Slim constructor validation" begin
    # Valid Slim span=0: 5 rows.
    rep0 = Matrix{Float32}(undef, 5, 4)
    rep0 .= 0.0
    @test_nowarn Slim("test", rep0, 0, 4)

    # Valid Slim span=1: 25 rows.
    rep1 = Matrix{Float32}(undef, 25, 4)
    rep1 .= 0.0
    @test_nowarn Slim("test", rep1, 1, 4)

    # Negative span.
    @test_throws ModelDimensionError Slim("test", rep0, -1, 4)

    # Span too high.
    @test_throws ModelDimensionError Slim("test", rep0, 11, 4)

    # Wrong row count for span.
    @test_throws ModelDimensionError Slim("test", Matrix{Float32}(undef, 10, 4), 1, 4)

    # Non-finite values.
    rep_bad = copy(rep0)
    rep_bad[1, 1] = NaN
    @test_throws ModelFormatError Slim("test", rep_bad, 0, 4)
end

# ── B2: extract_site_matrix bounds check ──────────────────────────────────

@testset "B2: extract_site_matrix rejects out-of-bounds windows" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    batch = EncodedSequenceBatch([encode_sequence("ACGTACGTACGT")])
    # Create a SiteCollection with an out-of-bounds start.
    # start=10 with motif_width=12 exceeds the 12-byte sequence at offset 0.
    coll = SiteCollection(
        Int[1],           # seq_indices
        Int[10],          # starts (1-based scan position)
        Int8[0],          # strands (forward)
        Float32[1.0],     # scores
    )
    @test_throws InvariantError extract_site_matrix(batch, coll, 12)
end
