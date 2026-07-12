# SiteGA dinucleotide scanning kernel.
#
# Generalizes the higher-order scan for the SiteGA geometry:
#   kmer = 2 (dinucleotide), context = 0, n_terms = motif_length - 1,
#   window_size = motif_length, scan_positions = seq_len - motif_length + 1.
#
# Forward scan at position `pos` (0-indexed relative to window start):
#   For term t (0..n_terms-1):
#     code = encode_5ary(seq[pos + t], seq[pos + t + 1])
#     score += representation[code, t]
#
# Reverse scan at position `pos`:
#   For term t:
#     code = encode_5ary(complement(seq[pos + window - 1 - t]),
#                        complement(seq[pos + window - 1 - (t + 1)]))
#     score += representation[code, t]

"""
    npositions_sitega(seq_len::Int, model::SiteGA)

Return the number of scanning positions for a SiteGA model.
"""
function npositions_sitega(seq_len::Int, model::SiteGA)
    return max(seq_len - window_size(model) + 1, 0)
end

# ── Forward scan kernel ──────────────────────────────────────────────────

"""
    scan_forward!(dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with forward-strand SiteGA scores for one sequence.
"""
function scan_forward!(
    dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
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
    scan_reverse!(dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with reverse-strand SiteGA scores for one sequence.
"""
function scan_reverse!(
    dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
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
    scan_best!(dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with the maximum of forward and reverse strand scores.
"""
function scan_best!(
    dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
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
    scan_both!(fwd::AbstractVector{T}, rev::AbstractVector{T}, model::SiteGA,
               seq::AbstractVector{UInt8}, n_pos::Int)

Fill `fwd` and `rev` with forward and reverse strand scores respectively.
"""
function scan_both!(
    fwd::AbstractVector{T},
    rev::AbstractVector{T},
    model::SiteGA,
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
    scan(model::SiteGA, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())

Scan a single encoded sequence with a [`SiteGA`](@ref) model.

Returns:
- `Vector{Float32}` for `ForwardOnly`, `ReverseOnly`, `BestStrand`.
- [`StrandPair{Vector{Float32}}`](@ref) for `BothStrands`.
"""
function scan(
    model::SiteGA, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly()
)
    n_pos = npositions_sitega(length(seq), model)
    return _scan_single_sitega(strands, model, seq, n_pos)
end

function _scan_single_sitega(
    ::ForwardOnly, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_single_sitega(
    ::ReverseOnly, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_single_sitega(
    ::BestStrand, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_single_sitega(
    ::BothStrands, model::SiteGA, seq::AbstractVector{UInt8}, n_pos::Int
)
    fwd = Vector{Float32}(undef, n_pos)
    rev = Vector{Float32}(undef, n_pos)
    scan_both!(fwd, rev, model, seq, n_pos)
    return StrandPair(fwd, rev)
end

# ── Single-sequence in-place scan ────────────────────────────────────────

"""
    scan!(dest::AbstractVector{T}, model::SiteGA, seq::AbstractVector{UInt8};
          strands::StrandPolicy=ForwardOnly())

Fill `dest` with scan scores for one sequence.
"""
function scan!(
    dest::AbstractVector{T},
    model::SiteGA,
    seq::AbstractVector{UInt8};
    strands::StrandPolicy=ForwardOnly(),
) where {T<:AbstractFloat}
    n_pos = npositions_sitega(length(seq), model)
    if length(dest) < n_pos
        throw(
            ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
        )
    end
    return _scan_inplace_sitega!(strands, dest, model, seq, n_pos)
end

function _scan_inplace_sitega!(
    ::ForwardOnly, dest::AbstractVector{T}, model::SiteGA, seq, n_pos
) where {T<:AbstractFloat}
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_inplace_sitega!(
    ::ReverseOnly, dest::AbstractVector{T}, model::SiteGA, seq, n_pos
) where {T<:AbstractFloat}
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_inplace_sitega!(
    ::BestStrand, dest::AbstractVector{T}, model::SiteGA, seq, n_pos
) where {T<:AbstractFloat}
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_inplace_sitega!(
    ::BothStrands, dest::AbstractVector{T}, model::SiteGA, seq, n_pos
) where {T<:AbstractFloat}
    return throw(
        ArgumentError(
            "scan! with BothStrands is not supported; use scan(model, seq; strands=BothStrands()).",
        ),
    )
end

# ── Batch scanning (EncodedSequenceBatch) ─────────────────────────────────

"""
    scan(model::SiteGA, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly())

Scan all sequences in a batch with a [`SiteGA`](@ref) model, returning a
[`RaggedArray{Float32}`](@ref) of scores.

For `BothStrands`, returns a [`StrandPair{RaggedArray{Float32}}`](@ref).
"""
function scan(
    model::SiteGA, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly()
)
    return _scan_batch_sitega(strands, model, batch)
end

function _scan_batch_sitega(
    strands::StrandPolicy, model::SiteGA, batch::EncodedSequenceBatch
)
    n = nsequences(batch)
    T = Float32

    out_rows = Vector{Vector{T}}(undef, n)
    for i in 1:n
        n_pos = npositions_sitega(seqlength(batch, i), model)
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

function _scan_batch_sitega(::BothStrands, model::SiteGA, batch::EncodedSequenceBatch)
    n = nsequences(batch)
    T = Float32

    fwd_rows = Vector{Vector{T}}(undef, n)
    rev_rows = Vector{Vector{T}}(undef, n)
    for i in 1:n
        n_pos = npositions_sitega(seqlength(batch, i), model)
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
    scan_result_lengths(model::SiteGA, batch::EncodedSequenceBatch)

Return a `Vector{Int}` with the number of scan positions for each sequence.
"""
function scan_result_lengths(model::SiteGA, batch::EncodedSequenceBatch)
    return [npositions_sitega(seqlength(batch, i), model) for i in 1:nsequences(batch)]
end
