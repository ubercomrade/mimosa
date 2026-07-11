using Test
using Mimosa

const REPO_ROOT = joinpath(dirname(dirname(@__DIR__)), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

@testset "reverse_complement involution (PWM)" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    rc = reverse_complement(pwm)
    @test reverse_complement(rc) == pwm
end

@testset "reverse_complement involution (PFM)" begin
    pfm = read_meme(joinpath(EXAMPLES, "pif4.meme"))
    rc = reverse_complement(pfm)
    @test reverse_complement(rc) == pfm
end

@testset "identical motif comparison gives pcc 1.0" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    @test compare(pwm, pwm; metric="pcc").score ≈ 1.0f0
end

@testset "non-! functions do not mutate inputs" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    copy_w = copy(pwm.weights)
    compare(pwm, pwm; metric="pcc")
    scorebounds(pwm)
    reverse_complement(pwm)
    @test pwm.weights == copy_w
end

@testset "comparison is deterministic" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    r1 = compare(pwm, pwm; metric="pcc")
    r2 = compare(pwm, pwm; metric="pcc")
    @test r1.score == r2.score
    @test r1.offset == r2.offset
    @test r1.orientation == r2.orientation
end

@testset "score bounds are consistent" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    mn, mx = scorebounds(pwm)
    @test mn <= mx
    @test isfinite(mn) && isfinite(mx)
end

@testset "orientation labels are valid" begin
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "gata2.meme"))
    for m in ("pcc", "ed", "cosine")
        r = compare(pwm1, pwm2; metric=m)
        @test r.orientation in ("++", "+-", "-+", "--")
    end
end

# Stage 2 properties

@testset "reverse_complement involution (encoded sequence)" begin
    for s in ["ACGT", "AAAA", "ACGTACGT", "NNNN", "", "ACGTNNACGT"]
        seq = encode_sequence(s)
        @test reverse_complement(reverse_complement(seq)) == seq
    end
end

@testset "scan does not mutate inputs" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    seq_copy = copy(seq)
    weights_copy = copy(pwm.weights)
    scan(pwm, seq; strands=ForwardOnly())
    scan(pwm, seq; strands=BothStrands())
    @test seq == seq_copy
    @test pwm.weights == weights_copy
end

@testset "scan is deterministic" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    r1 = scan(pwm, seq; strands=ForwardOnly())
    r2 = scan(pwm, seq; strands=ForwardOnly())
    @test r1 == r2
    p1 = scan(pwm, seq; strands=BothStrands())
    p2 = scan(pwm, seq; strands=BothStrands())
    @test p1.forward == p2.forward
    @test p1.reverse == p2.reverse
end

@testset "scan! == scan (forward)" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    for s in ["ACGTACGTACGTACGTACGTACGTAC", "GGGGCCCCAAAATTTTGGGGCCCCAA"]
        seq = encode_sequence(s)
        alloc = scan(pwm, seq; strands=ForwardOnly())
        dest = Vector{Float32}(undef, length(alloc))
        scan!(dest, pwm, seq; strands=ForwardOnly())
        @test dest == alloc
    end
end

@testset "scan! == scan (reverse)" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    alloc = scan(pwm, seq; strands=ReverseOnly())
    dest = Vector{Float32}(undef, length(alloc))
    scan!(dest, pwm, seq; strands=ReverseOnly())
    @test dest == alloc
end

@testset "scan! == scan (best)" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    alloc = scan(pwm, seq; strands=BestStrand())
    dest = Vector{Float32}(undef, length(alloc))
    scan!(dest, pwm, seq; strands=BestStrand())
    @test dest == alloc
end

@testset "best strand = max of fwd and rev" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    for s in ["ACGTACGTACGTACGTACGTACGTAC", "TTTTGGGGCCCCAAAATTTTGGGGCCC"]
        seq = encode_sequence(s)
        fwd = scan(pwm, seq; strands=ForwardOnly())
        rev = scan(pwm, seq; strands=ReverseOnly())
        best = scan(pwm, seq; strands=BestStrand())
        @test best == max.(fwd, rev)
    end
end

@testset "batch scan == single scan" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    rows = [
        encode_sequence("ACGTACGTACGTACGTACGTACGTAC"),
        encode_sequence("TTTTGGGGCCCCAAAATTTTGGGGCCC"),
        encode_sequence("ACGT"),
    ]
    batch = EncodedSequenceBatch(rows)
    fwd = scan(pwm, batch; strands=ForwardOnly())
    for i in 1:nsequences(batch)
        single = scan(pwm, sequence(batch, i); strands=ForwardOnly())
        @test row(fwd, i) == single
    end
    rev = scan(pwm, batch; strands=ReverseOnly())
    for i in 1:nsequences(batch)
        single = scan(pwm, sequence(batch, i); strands=ReverseOnly())
        @test row(rev, i) == single
    end
end

@testset "short sequence returns empty scores" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    W = length(pwm)
    for len in 0:(W - 1)
        seq = encode_sequence("A"^len)
        for strands in (ForwardOnly(), ReverseOnly(), BestStrand())
            @test scan(pwm, seq; strands=strands) == Float32[]
        end
        pair = scan(pwm, seq; strands=BothStrands())
        @test pair.forward == Float32[]
        @test pair.reverse == Float32[]
    end
    # Exactly motif width → 1 position
    seq = encode_sequence("A"^W)
    @test length(scan(pwm, seq; strands=ForwardOnly())) == 1
end

@testset "reverse scan equals forward scan of reverse-complement PWM" begin
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    rev_scores = scan(pwm, seq; strands=ReverseOnly())
    rc_pwm = reverse_complement(pwm)
    fwd_rc_scores = scan(rc_pwm, seq; strands=ForwardOnly())
    @test rev_scores ≈ fwd_rc_scores
end

@testset "batch reverse_complement involution" begin
    rows = [encode_sequence("ACGTACGT"), encode_sequence("TTTTGGGG"), encode_sequence("")]
    batch = EncodedSequenceBatch(rows)
    rc_batch = reverse_complement(batch)
    rc_rc_batch = reverse_complement(rc_batch)
    @test rc_rc_batch == batch
end

@testset "FASTA round-trip to_padded" begin
    batch, _ = read_fasta(joinpath(EXAMPLES, "foreground.fa"))
    padded, lengths = to_padded(batch)
    rt = from_padded(padded, lengths)
    @test rt == batch
end
