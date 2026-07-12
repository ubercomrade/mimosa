# Dimont scanning kernel.
#
# Dimont uses the same context-aware scanning geometry as BaMM:
#   kmer       = span + 1          (bases per scoring term)
#   context    = span              (bases before motif start used for context)
#   window     = motif_len + span  (total sequence window needed)
#   n_terms    = motif_len         (number of scoring terms per window)
#   n_positions = seq_len - window + 1
#
# The kernel is identical to BaMM's; only the model type and parameter names
# differ (span vs order).

"""
    npositions_dimont(seq_len::Int, model::Dimont)

Return the number of scanning positions for a Dimont model.
"""
function npositions_dimont(seq_len::Int, model::Dimont)
    return max(seq_len - window_size(model) + 1, 0)
end

# ── Forward scan kernel ──────────────────────────────────────────────────

"""
    scan_forward!(dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with forward-strand Dimont scores for one sequence.
"""
function scan_forward!(
    dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
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
    scan_reverse!(dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with reverse-strand Dimont scores for one sequence.
"""
function scan_reverse!(
    dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
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
    scan_best!(dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int)

Fill `dest[1:n_pos]` with the maximum of forward and reverse strand scores.
"""
function scan_best!(
    dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
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
    scan_both!(fwd::AbstractVector{T}, rev::AbstractVector{T}, model::Dimont,
               seq::AbstractVector{UInt8}, n_pos::Int)

Fill `fwd` and `rev` with forward and reverse strand scores respectively.
"""
function scan_both!(
    fwd::AbstractVector{T},
    rev::AbstractVector{T},
    model::Dimont,
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
    scan(model::Dimont, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())

Scan a single encoded sequence with a [`Dimont`](@ref) model.

Returns:
- `Vector{Float32}` for `ForwardOnly`, `ReverseOnly`, `BestStrand`.
- [`StrandPair{Vector{Float32}}`](@ref) for `BothStrands`.
"""
function scan(
    model::Dimont, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly()
)
    n_pos = npositions_dimont(length(seq), model)
    return _scan_single_dimont(strands, model, seq, n_pos)
end

function _scan_single_dimont(
    ::ForwardOnly, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_single_dimont(
    ::ReverseOnly, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_single_dimont(
    ::BestStrand, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_single_dimont(
    ::BothStrands, model::Dimont, seq::AbstractVector{UInt8}, n_pos::Int
)
    fwd = Vector{Float32}(undef, n_pos)
    rev = Vector{Float32}(undef, n_pos)
    scan_both!(fwd, rev, model, seq, n_pos)
    return StrandPair(fwd, rev)
end

# ── Single-sequence in-place scan ────────────────────────────────────────

"""
    scan!(dest::AbstractVector{T}, model::Dimont, seq::AbstractVector{UInt8};
          strands::StrandPolicy=ForwardOnly())

Fill `dest` with scan scores for one sequence.
"""
function scan!(
    dest::AbstractVector{T},
    model::Dimont,
    seq::AbstractVector{UInt8};
    strands::StrandPolicy=ForwardOnly(),
) where {T<:AbstractFloat}
    n_pos = npositions_dimont(length(seq), model)
    if length(dest) < n_pos
        throw(
            ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
        )
    end
    return _scan_inplace_dimont!(strands, dest, model, seq, n_pos)
end

function _scan_inplace_dimont!(
    ::ForwardOnly, dest::AbstractVector{T}, model::Dimont, seq, n_pos
) where {T<:AbstractFloat}
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_inplace_dimont!(
    ::ReverseOnly, dest::AbstractVector{T}, model::Dimont, seq, n_pos
) where {T<:AbstractFloat}
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_inplace_dimont!(
    ::BestStrand, dest::AbstractVector{T}, model::Dimont, seq, n_pos
) where {T<:AbstractFloat}
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_inplace_dimont!(
    ::BothStrands, dest::AbstractVector{T}, model::Dimont, seq, n_pos
) where {T<:AbstractFloat}
    return throw(
        ArgumentError(
            "scan! with BothStrands is not supported; use scan(model, seq; strands=BothStrands()).",
        ),
    )
end

# ── Batch scanning (EncodedSequenceBatch) ─────────────────────────────────

"""
    scan(model::Dimont, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly())

Scan all sequences in a batch with a [`Dimont`](@ref) model, returning a
[`RaggedArray{Float32}`](@ref) of scores.

For `BothStrands`, returns a [`StrandPair{RaggedArray{Float32}}`](@ref).
"""
function scan(
    model::Dimont, batch::EncodedSequenceBatch; strands::StrandPolicy=ForwardOnly()
)
    return _scan_batch_dimont(strands, model, batch)
end

function _scan_batch_dimont(
    strands::StrandPolicy, model::Dimont, batch::EncodedSequenceBatch
)
    n = nsequences(batch)
    T = Float32

    out_rows = Vector{Vector{T}}(undef, n)
    for i in 1:n
        n_pos = npositions_dimont(seqlength(batch, i), model)
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

function _scan_batch_dimont(::BothStrands, model::Dimont, batch::EncodedSequenceBatch)
    n = nsequences(batch)
    T = Float32

    fwd_rows = Vector{Vector{T}}(undef, n)
    rev_rows = Vector{Vector{T}}(undef, n)
    for i in 1:n
        n_pos = npositions_dimont(seqlength(batch, i), model)
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
    scan_result_lengths(model::Dimont, batch::EncodedSequenceBatch)

Return a `Vector{Int}` with the number of scan positions for each sequence.
"""
function scan_result_lengths(model::Dimont, batch::EncodedSequenceBatch)
    return [npositions_dimont(seqlength(batch, i), model) for i in 1:nsequences(batch)]
end
