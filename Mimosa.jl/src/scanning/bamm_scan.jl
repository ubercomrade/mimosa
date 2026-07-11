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
    kmer_val = kmer(model)
    ctx = context_length(model)
    n_terms = model.motif_length
    rep = model.representation
    seq_len = length(seq)

    @inbounds for pos in 1:n_pos
        total = zero(T)
        for term in 0:(n_terms - 1)
            code = 0
            src_start = pos - 1 - ctx + term  # 0-indexed source start
            for offset in 0:(kmer_val - 1)
                src = src_start + offset
                if 0 <= src < seq_len
                    encoded = Int(seq[src + 1])
                else
                    encoded = 4
                end
                code = code * 5 + encoded
            end
            total += rep[code + 1, term + 1]
        end
        dest[pos] = total
    end
    return dest
end

# ── Reverse scan kernel ──────────────────────────────────────────────────

"""
    scan_reverse!(dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with reverse-strand BaMM scores for one sequence.
"""
function scan_reverse!(
    dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    kmer_val = kmer(model)
    win = window_size(model)
    n_terms = model.motif_length
    rep = model.representation
    seq_len = length(seq)

    @inbounds for pos in 1:n_pos
        total = zero(T)
        for term in 0:(n_terms - 1)
            code = 0
            for offset in 0:(kmer_val - 1)
                src = (pos - 1) + (win - 1) - (term + offset)  # 0-indexed
                if 0 <= src < seq_len
                    base = Int(seq[src + 1])
                    encoded = base == 4 ? 4 : 3 - base
                else
                    encoded = 4
                end
                code = code * 5 + encoded
            end
            total += rep[code + 1, term + 1]
        end
        dest[pos] = total
    end
    return dest
end

# ── Best-strand scan kernel ──────────────────────────────────────────────

"""
    scan_best!(dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with the maximum of forward and reverse strand scores.
"""
function scan_best!(
    dest::AbstractVector{T}, model::BaMM, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    kmer_val = kmer(model)
    ctx = context_length(model)
    win = window_size(model)
    n_terms = model.motif_length
    rep = model.representation
    seq_len = length(seq)

    @inbounds for pos in 1:n_pos
        fwd_total = zero(T)
        rev_total = zero(T)
        for term in 0:(n_terms - 1)
            # Forward
            fwd_code = 0
            src_start = (pos - 1) - ctx + term
            for offset in 0:(kmer_val - 1)
                src = src_start + offset
                if 0 <= src < seq_len
                    encoded = Int(seq[src + 1])
                else
                    encoded = 4
                end
                fwd_code = fwd_code * 5 + encoded
            end
            fwd_total += rep[fwd_code + 1, term + 1]

            # Reverse
            rev_code = 0
            for offset in 0:(kmer_val - 1)
                src = (pos - 1) + (win - 1) - (term + offset)
                if 0 <= src < seq_len
                    base = Int(seq[src + 1])
                    encoded = base == 4 ? 4 : 3 - base
                else
                    encoded = 4
                end
                rev_code = rev_code * 5 + encoded
            end
            rev_total += rep[rev_code + 1, term + 1]
        end
        dest[pos] = max(fwd_total, rev_total)
    end
    return dest
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
    kmer_val = kmer(model)
    ctx = context_length(model)
    win = window_size(model)
    n_terms = model.motif_length
    rep = model.representation
    seq_len = length(seq)

    @inbounds for pos in 1:n_pos
        fwd_total = zero(T)
        rev_total = zero(T)
        for term in 0:(n_terms - 1)
            # Forward
            fwd_code = 0
            src_start = (pos - 1) - ctx + term
            for offset in 0:(kmer_val - 1)
                src = src_start + offset
                if 0 <= src < seq_len
                    encoded = Int(seq[src + 1])
                else
                    encoded = 4
                end
                fwd_code = fwd_code * 5 + encoded
            end
            fwd_total += rep[fwd_code + 1, term + 1]

            # Reverse
            rev_code = 0
            for offset in 0:(kmer_val - 1)
                src = (pos - 1) + (win - 1) - (term + offset)
                if 0 <= src < seq_len
                    base = Int(seq[src + 1])
                    encoded = base == 4 ? 4 : 3 - base
                else
                    encoded = 4
                end
                rev_code = rev_code * 5 + encoded
            end
            rev_total += rep[rev_code + 1, term + 1]
        end
        fwd[pos] = fwd_total
        rev[pos] = rev_total
    end
    return (fwd, rev)
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
    scan(model::BaMM, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly())

Scan all sequences in a batch with a [`BaMM`](@ref) model, returning a
[`RaggedArray{Float32}`](@ref) of scores.

For `BothStrands`, returns a [`StrandPair{RaggedArray{Float32}}`](@ref).
"""
function scan(model::BaMM, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly())
    return _scan_batch_bamm(strands, model, batch)
end

function _scan_batch_bamm(strands::StrandPolicy, model::BaMM, batch::EncodedSequenceBatch)
    n = nsequences(batch)
    T = Float32

    out_rows = Vector{Vector{T}}(undef, n)
    for i in 1:n
        n_pos = npositions_bamm(seqlength(batch, i), model)
        out_rows[i] = Vector{T}(undef, n_pos)
    end

    if strands isa ForwardOnly
        for i in 1:n
            scan_forward!(out_rows[i], model, sequence(batch, i), length(out_rows[i]))
        end
    elseif strands isa ReverseOnly
        for i in 1:n
            scan_reverse!(out_rows[i], model, sequence(batch, i), length(out_rows[i]))
        end
    elseif strands isa BestStrand
        for i in 1:n
            scan_best!(out_rows[i], model, sequence(batch, i), length(out_rows[i]))
        end
    end

    return build_ragged(out_rows)
end

function _scan_batch_bamm(::BothStrands, model::BaMM, batch::EncodedSequenceBatch)
    n = nsequences(batch)
    T = Float32

    fwd_rows = Vector{Vector{T}}(undef, n)
    rev_rows = Vector{Vector{T}}(undef, n)
    for i in 1:n
        n_pos = npositions_bamm(seqlength(batch, i), model)
        fwd_rows[i] = Vector{T}(undef, n_pos)
        rev_rows[i] = Vector{T}(undef, n_pos)
    end

    for i in 1:n
        scan_both!(fwd_rows[i], rev_rows[i], model, sequence(batch, i), length(fwd_rows[i]))
    end

    return StrandPair(build_ragged(fwd_rows), build_ragged(rev_rows))
end

# ── Scan result lengths ─────────────────────────────────────────────────

"""
    scan_result_lengths(model::BaMM, batch::EncodedSequenceBatch)

Return a `Vector{Int}` with the number of scan positions for each sequence.
"""
function scan_result_lengths(model::BaMM, batch::EncodedSequenceBatch)
    return [npositions_bamm(seqlength(batch, i), model) for i in 1:nsequences(batch)]
end
