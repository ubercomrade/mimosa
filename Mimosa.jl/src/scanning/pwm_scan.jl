# PWM scanning kernels: forward, reverse, best, and both strands.
#
# For a PWM with weights[base, position] (base 1:5 = A,C,G,T,N), the forward
# score at position `pos` in sequence `seq` (0-indexed: 0=A,1=C,2=G,3=T,4=N)
# is: sum_{p=1..W} weights[seq[pos+p-1] + 1, p]
#
# The reverse score at position `pos` is equivalent to scoring the window
# backwards with base complements:
# sum_{p=1..W} weights[complement(seq[pos+W-p]) + 1, p]
# where complement(b) = 4 if b==4, else 3-b.

"""
    npositions(seq_len::Int, motif_width::Int)

Return the number of scanning positions: max(seq_len - motif_width + 1, 0).
"""
function npositions(seq_len::Int, motif_width::Int)
    seq_len < 0 && throw(ArgumentError("sequence length must be non-negative."))
    motif_width < 1 && throw(ArgumentError("motif width must be positive."))
    return max(seq_len - motif_width + 1, 0)
end

npositions(model::PWM, seq_len::Int) = npositions(seq_len, motif_length(model))
kmer(::PWM) = 1
context_length(::PWM) = 0
scan_width(model::PWM) = motif_length(model)

function scan(model::PFM, args...; kwargs...)
    return throw(
        ArgumentError("PFM is not directly scannable; convert it with pwm_from_pfm first.")
    )
end

# Public scan entry points accept raw vectors, so validate once before entering
# the @inbounds kernels. Batch containers are already validated at construction.
function _validate_scan_input(seq::AbstractVector{UInt8}, n_pos::Int, width::Int, dests...)
    Base.require_one_based_indexing(seq)
    for dest in dests
        Base.require_one_based_indexing(dest)
    end
    n_pos < 0 && throw(ArgumentError("n_pos must be non-negative, got $n_pos."))
    width < 1 && throw(ArgumentError("scan width must be positive."))
    n_pos > npositions(length(seq), width) &&
        throw(ArgumentError("n_pos=$n_pos exceeds sequence geometry for width=$width."))
    any(code -> code > N_CODE, seq) && throw(
        ArgumentError(
            "sequence contains an invalid encoded DNA code; valid codes are 0x00..0x04."
        ),
    )
    any(length(dest) < n_pos for dest in dests) &&
        throw(ArgumentError("destination is shorter than n_pos=$n_pos."))
    return nothing
end

# ── Forward scan kernel ──────────────────────────────────────────────────

"""
    scan_forward!(dest::AbstractVector{Float32}, weights::AbstractMatrix{Float32},
                  seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with forward-strand PWM scores for one sequence.
The `weights` matrix has shape (5, W) with rows A,C,G,T,N.
"""
function scan_forward!(
    dest::AbstractVector{T},
    weights::AbstractMatrix{T},
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    W = size(weights, 2)
    size(weights, 1) == 5 || throw(ArgumentError("PWM weights must have 5 rows."))
    _validate_scan_input(seq, n_pos, W, dest)
    # Invariant: seq codes ∈ 0..N_CODE (guaranteed by EncodedSequenceBatch).
    # weights has 5 rows, so Int(seq[i])+1 ∈ 1..5 is always in bounds.
    # @inbounds is safe: pos ranges 1..n_pos, p ranges 1..W, and
    # pos+p-1 ≤ n_pos+W-1 ≤ length(seq) (since n_pos = length(seq)-W+1).
    @inbounds for pos in 1:n_pos
        total = zero(T)
        for p in 1:W
            base = Int(seq[pos + p - 1]) + 1
            total += weights[base, p]
        end
        dest[pos] = total
    end
    return dest
end

# ── Reverse scan kernel ──────────────────────────────────────────────────

"""
    scan_reverse!(dest::AbstractVector{Float32}, weights::AbstractMatrix{Float32},
                  seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with reverse-strand PWM scores for one sequence.
This is equivalent to scanning with the reverse-complement PWM on the forward strand.
"""
function scan_reverse!(
    dest::AbstractVector{T},
    weights::AbstractMatrix{T},
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    W = size(weights, 2)
    size(weights, 1) == 5 || throw(ArgumentError("PWM weights must have 5 rows."))
    _validate_scan_input(seq, n_pos, W, dest)
    # Invariant: seq codes ∈ 0..N_CODE; complement(b) = N_CODE or 0x03-b,
    # which is also ∈ 0..N_CODE.  @inbounds safe: pos+W-p ≥ pos+1-1 = pos ≥ 1,
    # and pos+W-p ≤ n_pos+W-1 ≤ length(seq).
    @inbounds for pos in 1:n_pos
        total = zero(T)
        for p in 1:W
            b = seq[pos + W - p]
            comp = b == N_CODE ? N_CODE : 0x03 - b
            total += weights[comp + 1, p]
        end
        dest[pos] = total
    end
    return dest
end

# ── Best-strand scan kernel ──────────────────────────────────────────────

"""
    scan_best!(dest::AbstractVector{Float32}, weights::AbstractMatrix{Float32},
               seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with the maximum of forward and reverse strand scores
at each position.
"""
function scan_best!(
    dest::AbstractVector{T},
    weights::AbstractMatrix{T},
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    W = size(weights, 2)
    size(weights, 1) == 5 || throw(ArgumentError("PWM weights must have 5 rows."))
    _validate_scan_input(seq, n_pos, W, dest)
    # Invariant: same as scan_forward! and scan_reverse! above.
    @inbounds for pos in 1:n_pos
        fwd = zero(T)
        rev = zero(T)
        for p in 1:W
            base = Int(seq[pos + p - 1]) + 1
            fwd += weights[base, p]
            b = seq[pos + W - p]
            comp = b == N_CODE ? N_CODE : 0x03 - b
            rev += weights[comp + 1, p]
        end
        dest[pos] = max(fwd, rev)
    end
    return dest
end

# ── Both-strand scan kernel ──────────────────────────────────────────────

"""
    scan_both!(fwd::AbstractVector{Float32}, rev::AbstractVector{Float32},
               weights::AbstractMatrix{Float32}, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `fwd[1:n_pos]` with forward scores and `rev[1:n_pos]` with reverse scores.
Both buffers must have at least `n_pos` elements.
"""
function scan_both!(
    fwd::AbstractVector{T},
    rev::AbstractVector{T},
    weights::AbstractMatrix{T},
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    W = size(weights, 2)
    size(weights, 1) == 5 || throw(ArgumentError("PWM weights must have 5 rows."))
    Base.mightalias(fwd, rev) &&
        throw(ArgumentError("forward and reverse destinations must not alias."))
    _validate_scan_input(seq, n_pos, W, fwd, rev)
    # Invariant: same as scan_forward! and scan_reverse! above.
    @inbounds for pos in 1:n_pos
        fwd_total = zero(T)
        rev_total = zero(T)
        for p in 1:W
            base = Int(seq[pos + p - 1]) + 1
            fwd_total += weights[base, p]
            b = seq[pos + W - p]
            comp = b == N_CODE ? N_CODE : 0x03 - b
            rev_total += weights[comp + 1, p]
        end
        fwd[pos] = fwd_total
        rev[pos] = rev_total
    end
    return (fwd, rev)
end

# ── Single-sequence allocating scan ──────────────────────────────────────

"""
    scan(model::PWM, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())

Scan a single encoded sequence with a [`PWM`](@ref) model.

Returns:
- `Vector{Float32}` for `ForwardOnly`, `ReverseOnly`, `BestStrand`.
- [`StrandPair{Vector{Float32}}`](@ref) for `BothStrands`.

Sequences shorter than the motif width return an empty score vector.
"""
function scan(model::PWM, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())
    n_pos = npositions(model, length(seq))
    return _scan_single(strands, model, seq, n_pos)
end

function _scan_single(::ForwardOnly, model::PWM, seq::AbstractVector{UInt8}, n_pos::Int)
    dest = Vector{Float32}(undef, n_pos)
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_single(::ReverseOnly, model::PWM, seq::AbstractVector{UInt8}, n_pos::Int)
    dest = Vector{Float32}(undef, n_pos)
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_single(::BestStrand, model::PWM, seq::AbstractVector{UInt8}, n_pos::Int)
    dest = Vector{Float32}(undef, n_pos)
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_single(::BothStrands, model::PWM, seq::AbstractVector{UInt8}, n_pos::Int)
    fwd = Vector{Float32}(undef, n_pos)
    rev = Vector{Float32}(undef, n_pos)
    scan_both!(fwd, rev, model, seq, n_pos)
    return StrandPair(fwd, rev)
end

# ── Single-sequence in-place scan ────────────────────────────────────────

"""
    scan!(dest::AbstractVector{Float32}, model::PWM, seq::AbstractVector{UInt8};
          strands::StrandPolicy=ForwardOnly())

Fill `dest` with scan scores for one sequence. The destination must have at
least `max(length(seq) - motif_width + 1, 0)` elements.

Supported strand policies: `ForwardOnly`, `ReverseOnly`, `BestStrand`.
For `BothStrands` use `scan(model, seq; strands=BothStrands())`.
"""
function scan!(
    dest::AbstractVector{T},
    model::PWM,
    seq::AbstractVector{UInt8};
    strands::StrandPolicy=ForwardOnly(),
) where {T<:AbstractFloat}
    n_pos = npositions(model, length(seq))
    if length(dest) < n_pos
        throw(
            ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
        )
    end
    return _scan_inplace!(strands, dest, model, seq, n_pos)
end

function _scan_inplace!(
    ::ForwardOnly, dest::AbstractVector{T}, model::PWM, seq, n_pos
) where {T<:AbstractFloat}
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_inplace!(
    ::ReverseOnly, dest::AbstractVector{T}, model::PWM, seq, n_pos
) where {T<:AbstractFloat}
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_inplace!(
    ::BestStrand, dest::AbstractVector{T}, model::PWM, seq, n_pos
) where {T<:AbstractFloat}
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_inplace!(
    ::BothStrands, dest::AbstractVector{T}, model::PWM, seq, n_pos
) where {T<:AbstractFloat}
    return throw(
        ArgumentError(
            "scan! with BothStrands is not supported; use scan(model, seq; strands=BothStrands()).",
        ),
    )
end

# ── Batch scanning (EncodedSequenceBatch) ─────────────────────────────────

"""
    scan(model::PWM, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly(),
         execution::ExecutionPolicy=SerialExecution())

Scan all sequences in a batch with a [`PWM`](@ref) model, returning a
[`RaggedArray{Float32}`](@ref) of scores.

For `BothStrands`, returns a [`StrandPair{RaggedArray{Float32}}`](@ref).

Under `ThreadedExecution`, sequences are processed in parallel at the
top level (one sequence per task chunk). Inner scanning kernels remain
serial. Results are identical to `SerialExecution` — they are written to
pre-allocated slots indexed by original position.
"""
function scan(
    model::PWM,
    batch::EncodedSequenceBatch;
    strands::StrandPolicy=ForwardOnly(),
    execution::ExecutionPolicy=SerialExecution(),
)
    return _scan_model_batch(model, batch; strands=strands, execution=execution)
end

function _scan_batch(
    strands::StrandPolicy, model::PWM, batch::EncodedSequenceBatch, ::SerialExecution
)
    n = nsequences(batch)
    W = length(model)
    weights = eltype(model.weights) === Float32 ? model.weights : Float32.(model.weights)
    T = Float32
    offsets = _scan_offsets(batch, W)
    data = Vector{T}(undef, offsets[end] - 1)
    for i in 1:n
        seq = sequence(batch, i)
        _scan_one_seq!(
            strands, _scan_dest(data, offsets, i), weights, seq, offsets[i + 1] - offsets[i]
        )
    end
    return RaggedArray(data, offsets)
end

function _scan_batch(
    strands::StrandPolicy, model::PWM, batch::EncodedSequenceBatch, pol::ThreadedExecution
)
    n = nsequences(batch)
    W = length(model)
    weights = eltype(model.weights) === Float32 ? model.weights : Float32.(model.weights)
    T = Float32

    offsets = _scan_offsets(batch, W)
    data = Vector{T}(undef, offsets[end] - 1)

    # Parallel execution: each task processes its chunk of sequences
    # Results written to pre-allocated slots → deterministic order
    _parallel_for_weighted(pol, _scan_costs(offsets)) do i
        seq = sequence(batch, i)
        return _scan_one_seq!(
            strands, _scan_dest(data, offsets, i), weights, seq, offsets[i + 1] - offsets[i]
        )
    end
    return RaggedArray(data, offsets)
end

# Dispatch helper: scan one sequence into pre-allocated dest
function _scan_one_seq!(
    ::ForwardOnly, dest::AbstractVector{T}, weights::AbstractMatrix{T}, seq, n_pos
) where {T<:AbstractFloat}
    return scan_forward!(dest, weights, seq, n_pos)
end
function _scan_one_seq!(
    ::ReverseOnly, dest::AbstractVector{T}, weights::AbstractMatrix{T}, seq, n_pos
) where {T<:AbstractFloat}
    return scan_reverse!(dest, weights, seq, n_pos)
end
function _scan_one_seq!(
    ::BestStrand, dest::AbstractVector{T}, weights::AbstractMatrix{T}, seq, n_pos
) where {T<:AbstractFloat}
    return scan_best!(dest, weights, seq, n_pos)
end

function _scan_batch(
    ::BothStrands, model::PWM, batch::EncodedSequenceBatch, ::SerialExecution
)
    n = nsequences(batch)
    W = length(model)
    weights = eltype(model.weights) === Float32 ? model.weights : Float32.(model.weights)
    T = Float32

    offsets = _scan_offsets(batch, W)
    fwd = Vector{T}(undef, offsets[end] - 1)
    rev = Vector{T}(undef, offsets[end] - 1)

    for i in 1:n
        seq = sequence(batch, i)
        scan_both!(
            _scan_dest(fwd, offsets, i),
            _scan_dest(rev, offsets, i),
            weights,
            seq,
            offsets[i + 1] - offsets[i],
        )
    end
    return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, copy(offsets)))
end

function _scan_batch(
    ::BothStrands, model::PWM, batch::EncodedSequenceBatch, pol::ThreadedExecution
)
    n = nsequences(batch)
    W = length(model)
    weights = eltype(model.weights) === Float32 ? model.weights : Float32.(model.weights)
    T = Float32

    offsets = _scan_offsets(batch, W)
    fwd = Vector{T}(undef, offsets[end] - 1)
    rev = Vector{T}(undef, offsets[end] - 1)

    _parallel_for_weighted(pol, _scan_costs(offsets)) do i
        seq = sequence(batch, i)
        return scan_both!(
            _scan_dest(fwd, offsets, i),
            _scan_dest(rev, offsets, i),
            weights,
            seq,
            offsets[i + 1] - offsets[i],
        )
    end
    return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, copy(offsets)))
end

function _scan_offsets(batch::EncodedSequenceBatch, width::Int)
    offsets = Vector{Int}(undef, nsequences(batch) + 1)
    offsets[1] = 1
    for i in 1:nsequences(batch)
        offsets[i + 1] = offsets[i] + npositions(seqlength(batch, i), width)
    end
    return offsets
end

function _scan_costs(offsets::Vector{Int})
    costs = Vector{Int}(undef, length(offsets) - 1)
    @inbounds for i in eachindex(costs)
        costs[i] = offsets[i + 1] - offsets[i]
    end
    return costs
end

function _scan_dest(data::AbstractVector, offsets::Vector{Int}, row_index::Int)
    start = offsets[row_index]
    stop = offsets[row_index + 1] - 1
    return start > stop ? view(data, 1:0) : @view(data[start:stop])
end

# ── Score bounds for scan results ─────────────────────────────────────────

"""
    scan_result_lengths(model::PWM, batch::EncodedSequenceBatch)

Return a `Vector{Int}` with the number of scan positions for each sequence
in the batch.
"""
function scan_result_lengths(model::PWM, batch::EncodedSequenceBatch)
    W = length(model)
    return [npositions(seqlength(batch, i), W) for i in 1:nsequences(batch)]
end
