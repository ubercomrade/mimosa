# Higher-order scanning kernel for BaMM and other context-aware motif models.
#
# Generalizes the PWM scan to handle k-mer context: each position score depends
# on `kmer` consecutive bases (kmer = order + 1), not just a single base.
#
# Scanning geometry:
#   kmer       = order + 1          (bases per scoring term)
#   context    = order              (bases before motif start used for context)
#   window     = motif_len + order  (total sequence window needed)
#   n_terms    = motif_len          (number of scoring terms per window)
#   n_positions = seq_len - window + 1
#
# Forward scan at position `pos` (0-indexed relative to window start):
#   For term t (0..n_terms-1):
#     code = encode_5ary(seq[pos - context + t], ..., seq[pos - context + t + kmer - 1])
#     score += representation[code, t + 1]
#
# Reverse scan at position `pos`:
#   For term t:
#     code = encode_5ary(complement(seq[pos + window - 1 - (t + 0)]),
#                        complement(seq[pos + window - 1 - (t + 1)]), ...)
#     score += representation[code, t + 1]

"""
    npositions_bamm(seq_len::Int, model::BaMM)

Return the number of scanning positions for a BaMM model.
"""
function npositions_bamm(seq_len::Int, model::BaMM)
    return max(seq_len - window_size(model) + 1, 0)
end

# ── Forward scan kernel ──────────────────────────────────────────────────

"""
    scan_forward!(dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with forward-strand BaMM scores for one sequence.
"""
function scan_forward!(
    dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    return _ho_scan_forward!(
        dest,
        model.representation,
        kmer(model),
        context_length(model),
        scan_width(model),
        seq,
        n_pos,
    )
end

# ── Reverse scan kernel ──────────────────────────────────────────────────

"""
    scan_reverse!(dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with reverse-strand BaMM scores for one sequence.
"""
function scan_reverse!(
    dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    return _ho_scan_reverse!(
        dest,
        model.representation,
        kmer(model),
        window_size(model),
        scan_width(model),
        seq,
        n_pos,
    )
end

# ── Best-strand scan kernel ──────────────────────────────────────────────

"""
    scan_best!(dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with the maximum of forward and reverse strand scores.
"""
function scan_best!(
    dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    return _ho_scan_best!(
        dest,
        model.representation,
        kmer(model),
        context_length(model),
        window_size(model),
        scan_width(model),
        seq,
        n_pos,
    )
end

# ── Both-strand scan kernel ──────────────────────────────────────────────

"""
    scan_both!(fwd::AbstractVector{T}, rev::AbstractVector{T}, model::BaMM,
               seq::AbstractVector{UInt8}, n_pos::Int)

Fill `fwd` and `rev` with forward and reverse strand scores respectively.
"""
function scan_both!(
    fwd::AbstractVector{T},
    rev::AbstractVector{T},
    model::BaMM,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    return _ho_scan_both!(
        fwd,
        rev,
        model.representation,
        kmer(model),
        context_length(model),
        window_size(model),
        scan_width(model),
        seq,
        n_pos,
    )
end

# ── Single-sequence allocating scan ──────────────────────────────────────

"""
    scan(model::BaMM, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())

Scan a single encoded sequence with a [`BaMM`](@ref) model.

Returns:
- `Vector{Float32}` for `ForwardOnly`, `ReverseOnly`, `BestStrand`.
- [`StrandPair{Vector{Float32}}`](@ref) for `BothStrands`.
"""
function scan(model::BaMM, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())
    n_pos = npositions_bamm(length(seq), model)
    return _scan_single_bamm(strands, model, seq, n_pos)
end

function _scan_single_bamm(
    ::ForwardOnly, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_single_bamm(
    ::ReverseOnly, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_single_bamm(
    ::BestStrand, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_single_bamm(
    ::BothStrands, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
)
    fwd = Vector{Float32}(undef, n_pos)
    rev = Vector{Float32}(undef, n_pos)
    scan_both!(fwd, rev, model, seq, n_pos)
    return StrandPair(fwd, rev)
end

# ── Single-sequence in-place scan ────────────────────────────────────────

"""
    scan!(dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8};
          strands::StrandPolicy=ForwardOnly())

Fill `dest` with scan scores for one sequence.
"""
function scan!(
    dest::AbstractVector{T},
    model::BaMM,
    seq::AbstractVector{UInt8};
    strands::StrandPolicy=ForwardOnly(),
) where {T<:AbstractFloat}
    n_pos = npositions_bamm(length(seq), model)
    if length(dest) < n_pos
        throw(
            ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
        )
    end
    return _scan_inplace_bamm!(strands, dest, model, seq, n_pos)
end

function _scan_inplace_bamm!(
    ::ForwardOnly, dest::AbstractVector{T}, model::BaMM, seq, n_pos
) where {T<:AbstractFloat}
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_inplace_bamm!(
    ::ReverseOnly, dest::AbstractVector{T}, model::BaMM, seq, n_pos
) where {T<:AbstractFloat}
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_inplace_bamm!(
    ::BestStrand, dest::AbstractVector{T}, model::BaMM, seq, n_pos
) where {T<:AbstractFloat}
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_inplace_bamm!(
    ::BothStrands, dest::AbstractVector{T}, model::BaMM, seq, n_pos
) where {T<:AbstractFloat}
    return throw(
        ArgumentError(
            "scan! with BothStrands is not supported; use scan(model, seq; strands=BothStrands()).",
        ),
    )
end

# ── Batch scanning (EncodedSequenceBatch) ─────────────────────────────────

"""
    scan(model::BaMM, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly(),
         execution::ExecutionPolicy=SerialExecution())

Scan all sequences in a batch with a [`BaMM`](@ref) model, returning a
[`RaggedArray{Float32}`](@ref) of scores.

For `BothStrands`, returns a [`StrandPair{RaggedArray{Float32}}`](@ref).

Under `ThreadedExecution`, sequences are processed in parallel at the
top level. Inner scanning kernels remain serial.
"""
function scan(
    model::BaMM,
    batch::EncodedSequenceBatch;
    strands::StrandPolicy=ForwardOnly(),
    execution::ExecutionPolicy=SerialExecution(),
)
    if strands isa BothStrands
        return _ho_scan_batch_both(
            model,
            batch,
            (sl, m) -> npositions_bamm(sl, m),
            (fwd, rev, m, seq, npos) -> scan_both!(fwd, rev, m, seq, npos),
            execution,
        )
    end
    scan_fn! = if strands isa ForwardOnly
        (dest, m, seq, npos) -> scan_forward!(dest, m, seq, npos)
    elseif strands isa ReverseOnly
        (dest, m, seq, npos) -> scan_reverse!(dest, m, seq, npos)
    else # BestStrand
        (dest, m, seq, npos) -> scan_best!(dest, m, seq, npos)
    end
    return _ho_scan_batch(
        strands, model, batch, (sl, m) -> npositions_bamm(sl, m), scan_fn!, execution
    )
end

# ── Scan result lengths ─────────────────────────────────────────────────

"""
    scan_result_lengths(model::BaMM, batch::EncodedSequenceBatch)

Return a `Vector{Int}` with the number of scan positions for each sequence.
"""
function scan_result_lengths(model::BaMM, batch::EncodedSequenceBatch)
    return [npositions_bamm(seqlength(batch, i), model) for i in 1:nsequences(batch)]
end
