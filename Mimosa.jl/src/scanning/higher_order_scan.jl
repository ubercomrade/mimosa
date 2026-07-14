# Generic context-aware (higher-order) scanning kernel shared by BaMM,
# Dimont, Slim and any model whose per-window score is the column-wise sum
# of a `(5^kmer, n_terms)` representation indexed by a 5-ary code of
# `kmer` consecutive (possibly complemented) bases.
#
# Geometry (provided by the concrete model via `kmer`, `context_length`,
# `window_size`, `scan_width`):
#   kmer_val   = bases per scoring term
#   ctx        = bases before motif start used for context
#   win        = total sequence window needed
#   n_terms    = number of scoring terms per window (= `scan_width(model)`)
#   n_pos      = number of scan positions = seq_len - win + 1
#
# Forward term t (0-indexed):
#   code = encode_5ary(seq[pos - ctx + t], ..., seq[pos - ctx + t + kmer - 1])
#   score += representation[code, t]
# Reverse term t:
#   code = encode_5ary(complement(seq[pos + win - 1 - (t + 0)]), ...)
#   score += representation[code, t]
#
# Out-of-window positions use the N encoding (4). This kernel is type-stable
# and allocates nothing in the inner loop.

"""
    _ho_scan_forward!(dest, rep, kmer_val, ctx, n_terms, seq, n_pos)

Fill `dest[1:n_pos]` with forward-strand scores for one sequence using the
generic higher-order kernel. `rep` is indexed `[code+1, term+1]`.
"""
function _ho_scan_forward!(
    dest::AbstractVector{T},
    rep::AbstractMatrix,
    kmer_val::Int,
    ctx::Int,
    n_terms::Int,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    n_pos < 0 && throw(ArgumentError("n_pos must be non-negative, got $n_pos."))
    length(dest) < n_pos && throw(
        ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
    )
    seq_len = length(seq)
    # Invariant: seq codes in 0..N_CODE (guaranteed by EncodedSequenceBatch).
    # code = base*5 + ... for kmer_val bases, each in 0..4, so code in 0..5^kmer_val-1.
    # rep has 5^kmer_val rows, so code+1 is always in bounds.
    # @inbounds is safe: pos ranges 1..n_pos, term ranges 0..n_terms-1,
    # offset ranges 0..kmer_val-1.  Out-of-window positions use code 4 (N).
    @inbounds for pos in 1:n_pos
        total = zero(T)
        for term in 0:(n_terms - 1)
            code = 0
            src_start = (pos - 1) - ctx + term
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

"""
    _ho_scan_reverse!(dest, rep, kmer_val, win, n_terms, seq, n_pos)

Fill `dest[1:n_pos]` with reverse-strand scores for one sequence using the
generic higher-order kernel.
"""
function _ho_scan_reverse!(
    dest::AbstractVector{T},
    rep::AbstractMatrix,
    kmer_val::Int,
    win::Int,
    n_terms::Int,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    n_pos < 0 && throw(ArgumentError("n_pos must be non-negative, got $n_pos."))
    length(dest) < n_pos && throw(
        ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
    )
    seq_len = length(seq)
    # Invariant: same as _ho_scan_forward! but with complement for reverse strand.
    # complement(b) = N_CODE if b==N_CODE else 3-b, still in 0..N_CODE.
    @inbounds for pos in 1:n_pos
        total = zero(T)
        for term in 0:(n_terms - 1)
            code = 0
            for offset in 0:(kmer_val - 1)
                src = (pos - 1) + (win - 1) - (term + offset)
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

"""
    _ho_scan_best!(dest, rep, kmer_val, ctx, win, n_terms, seq, n_pos)

Fill `dest[1:n_pos]` with the per-position maximum of forward and reverse
strand scores.
"""
function _ho_scan_best!(
    dest::AbstractVector{T},
    rep::AbstractMatrix,
    kmer_val::Int,
    ctx::Int,
    win::Int,
    n_terms::Int,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    n_pos < 0 && throw(ArgumentError("n_pos must be non-negative, got $n_pos."))
    length(dest) < n_pos && throw(
        ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
    )
    seq_len = length(seq)
    # Invariant: same as _ho_scan_forward! and _ho_scan_reverse! above.
    @inbounds for pos in 1:n_pos
        fwd_total = zero(T)
        rev_total = zero(T)
        for term in 0:(n_terms - 1)
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

"""
    _ho_scan_both!(fwd, rev, rep, kmer_val, ctx, win, n_terms, seq, n_pos)

Fill `fwd` and `rev` with forward and reverse strand scores respectively.
"""
function _ho_scan_both!(
    fwd::AbstractVector{T},
    rev::AbstractVector{T},
    rep::AbstractMatrix,
    kmer_val::Int,
    ctx::Int,
    win::Int,
    n_terms::Int,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    n_pos < 0 && throw(ArgumentError("n_pos must be non-negative, got $n_pos."))
    (length(fwd) < n_pos || length(rev) < n_pos) && throw(
        ArgumentError("fwd/rev destinations must each have at least $n_pos elements.")
    )
    seq_len = length(seq)
    # Invariant: same as _ho_scan_forward! and _ho_scan_reverse! above.
    @inbounds for pos in 1:n_pos
        fwd_total = zero(T)
        rev_total = zero(T)
        for term in 0:(n_terms - 1)
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

# ── Generic parallel batch scan helpers ─────────────────────────────────────
#
# These helpers are used by all higher-order models (BaMM, SiteGA, Dimont, Slim)
# to avoid duplicating the pre-allocate + parallel-for pattern. Each model
# provides its own `_scan_one_seq!` dispatch for single/strand types.

"""
    _ho_scan_batch(strands, model, batch, npos_fn, scan_fn!, ::SerialExecution)

Generic serial batch scan for higher-order models. `npos_fn(seq_len, model)`
returns the number of scan positions, and `scan_fn!(dest, model, seq, n_pos)`
fills the destination.
"""
function _ho_scan_batch(
    strands::StrandPolicy,
    model::AbstractMotifModel,
    batch::EncodedSequenceBatch,
    npos_fn::F,
    scan_fn!::G,
    ::SerialExecution,
) where {F,G}
    n = nsequences(batch)
    offsets = _ho_scan_offsets(batch, model, npos_fn)
    data = Vector{Float32}(undef, offsets[end] - 1)
    for i in 1:n
        dest = _scan_dest(data, offsets, i)
        scan_fn!(dest, model, sequence(batch, i), length(dest))
    end
    return RaggedArray(data, offsets)
end

"""
    _ho_scan_batch(strands, model, batch, npos_fn, scan_fn!, pol::ThreadedExecution)

Generic threaded batch scan for higher-order models. Pre-allocates output
slots, then processes sequences in parallel. Results are written to
pre-allocated slots indexed by original position, so output order matches
serial execution.
"""
function _ho_scan_batch(
    strands::StrandPolicy,
    model::AbstractMotifModel,
    batch::EncodedSequenceBatch,
    npos_fn::F,
    scan_fn!::G,
    pol::ThreadedExecution,
) where {F,G}
    n = nsequences(batch)
    offsets = _ho_scan_offsets(batch, model, npos_fn)
    data = Vector{Float32}(undef, offsets[end] - 1)

    _parallel_for_weighted(pol, _scan_costs(offsets)) do i
        dest = _scan_dest(data, offsets, i)
        return scan_fn!(dest, model, sequence(batch, i), length(dest))
    end

    return RaggedArray(data, offsets)
end

"""
    _ho_scan_batch_both(model, batch, npos_fn, both_fn!, ::SerialExecution)

Generic serial batch scan for BothStrands mode. `both_fn!(fwd, rev, model, seq, n_pos)`
fills both destinations.
"""
function _ho_scan_batch_both(
    model::AbstractMotifModel,
    batch::EncodedSequenceBatch,
    npos_fn::F,
    both_fn!::G,
    ::SerialExecution,
) where {F,G}
    n = nsequences(batch)
    offsets = _ho_scan_offsets(batch, model, npos_fn)
    fwd = Vector{Float32}(undef, offsets[end] - 1)
    rev = similar(fwd)
    for i in 1:n
        fwd_dest = _scan_dest(fwd, offsets, i)
        rev_dest = _scan_dest(rev, offsets, i)
        both_fn!(fwd_dest, rev_dest, model, sequence(batch, i), length(fwd_dest))
    end
    return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, copy(offsets)))
end

"""
    _ho_scan_batch_both(model, batch, npos_fn, both_fn!, pol::ThreadedExecution)

Generic threaded batch scan for BothStrands mode.
"""
function _ho_scan_batch_both(
    model::AbstractMotifModel,
    batch::EncodedSequenceBatch,
    npos_fn::F,
    both_fn!::G,
    pol::ThreadedExecution,
) where {F,G}
    n = nsequences(batch)
    offsets = _ho_scan_offsets(batch, model, npos_fn)
    fwd = Vector{Float32}(undef, offsets[end] - 1)
    rev = similar(fwd)
    _parallel_for_weighted(pol, _scan_costs(offsets)) do i
        fwd_dest = _scan_dest(fwd, offsets, i)
        rev_dest = _scan_dest(rev, offsets, i)
        return both_fn!(fwd_dest, rev_dest, model, sequence(batch, i), length(fwd_dest))
    end
    return StrandPair(RaggedArray(fwd, offsets), RaggedArray(rev, copy(offsets)))
end

function _ho_scan_offsets(batch::EncodedSequenceBatch, model, npos_fn::F) where {F}
    offsets = Vector{Int}(undef, nsequences(batch) + 1)
    offsets[1] = 1
    @inbounds for i in 1:nsequences(batch)
        offsets[i + 1] = offsets[i] + npos_fn(seqlength(batch, i), model)
    end
    return offsets
end

# ── Generic AbstractHigherOrderMotif adapter ─────────────────────────────────
#
# The following methods provide a single generic implementation of the scan
# adapter layer for all AbstractHigherOrderMotif subtypes (BaMM, SiteGA, Dimont,
# Slim). They replace the per-model boilerplate that previously existed in
# bamm_scan.jl, dimont_scan.jl, slim_scan.jl, and sitega_scan.jl.
#
# Each model provides the geometry via the trait functions:
#   kmer(model), context_length(model), window_size(model), scan_width(model)
# and stores its scoring matrix in the `representation` field.
# The inner kernels (_ho_scan_forward!, _ho_scan_reverse!, etc.) remain shared
# and type-stable because the `representation` matrix is passed as an argument.

"""
    npositions_ho(seq_len::Int, model::AbstractHigherOrderMotif)

Return the number of scanning positions for a higher-order model.
This is the generic version of the per-model `npositions_*` functions.
"""
function npositions_ho(seq_len::Int, model::AbstractHigherOrderMotif)
    return max(seq_len - window_size(model) + 1, 0)
end

motif_length(model::AbstractHigherOrderMotif) = model.motif_length
is_scannable(::AbstractHigherOrderMotif) = true
npositions(model::AbstractHigherOrderMotif, seq_len::Int) = npositions_ho(seq_len, model)
scorematrix(model::AbstractHigherOrderMotif) = model.representation
scoretype(model::AbstractHigherOrderMotif) = eltype(scorematrix(model))

# ── Generic single-sequence scan kernels ──────────────────────────────────

"""
    scan_forward!(dest, model::AbstractHigherOrderMotif, seq, n_pos)

Fill `dest[1:n_pos]` with forward-strand scores for one sequence.
Generic method for all higher-order models.
"""
function scan_forward!(
    dest::AbstractVector{T},
    model::AbstractHigherOrderMotif,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    _validate_scan_input(seq, n_pos, window_size(model), dest)
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

"""
    scan_reverse!(dest, model::AbstractHigherOrderMotif, seq, n_pos)

Fill `dest[1:n_pos]` with reverse-strand scores for one sequence.
Generic method for all higher-order models.
"""
function scan_reverse!(
    dest::AbstractVector{T},
    model::AbstractHigherOrderMotif,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    _validate_scan_input(seq, n_pos, window_size(model), dest)
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

"""
    scan_best!(dest, model::AbstractHigherOrderMotif, seq, n_pos)

Fill `dest[1:n_pos]` with the per-position maximum of forward and reverse scores.
Generic method for all higher-order models.
"""
function scan_best!(
    dest::AbstractVector{T},
    model::AbstractHigherOrderMotif,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    _validate_scan_input(seq, n_pos, window_size(model), dest)
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

"""
    scan_both!(fwd, rev, model::AbstractHigherOrderMotif, seq, n_pos)

Fill `fwd` and `rev` with forward and reverse strand scores respectively.
Generic method for all higher-order models.
"""
function scan_both!(
    fwd::AbstractVector{T},
    rev::AbstractVector{T},
    model::AbstractHigherOrderMotif,
    seq::AbstractVector{UInt8},
    n_pos::Int,
) where {T<:AbstractFloat}
    Base.mightalias(fwd, rev) &&
        throw(ArgumentError("forward and reverse destinations must not alias."))
    _validate_scan_input(seq, n_pos, window_size(model), fwd, rev)
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

# ── Generic single-sequence allocating scan ──────────────────────────────

"""
    scan(model::AbstractHigherOrderMotif, seq; strands)

Scan a single encoded sequence with a higher-order model.
Generic method for all AbstractHigherOrderMotif subtypes.

Returns:
- `Vector{Float32}` for `ForwardOnly`, `ReverseOnly`, `BestStrand`.
- [`StrandPair{Vector{Float32}}`](@ref) for `BothStrands`.
"""
function scan(
    model::AbstractHigherOrderMotif,
    seq::AbstractVector{UInt8};
    strands::StrandPolicy=ForwardOnly(),
)
    n_pos = npositions_ho(length(seq), model)
    return _scan_single_ho(strands, model, seq, n_pos)
end

function _scan_single_ho(
    ::ForwardOnly, model::AbstractHigherOrderMotif, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_single_ho(
    ::ReverseOnly, model::AbstractHigherOrderMotif, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_single_ho(
    ::BestStrand, model::AbstractHigherOrderMotif, seq::AbstractVector{UInt8}, n_pos::Int
)
    dest = Vector{Float32}(undef, n_pos)
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_single_ho(
    ::BothStrands, model::AbstractHigherOrderMotif, seq::AbstractVector{UInt8}, n_pos::Int
)
    fwd = Vector{Float32}(undef, n_pos)
    rev = Vector{Float32}(undef, n_pos)
    scan_both!(fwd, rev, model, seq, n_pos)
    return StrandPair(fwd, rev)
end

# ── Generic single-sequence in-place scan ──────────────────────────────────

"""
    scan!(dest, model::AbstractHigherOrderMotif, seq; strands)

Fill `dest` with scan scores for one sequence.
Generic method for all AbstractHigherOrderMotif subtypes.
"""
function scan!(
    dest::AbstractVector{T},
    model::AbstractHigherOrderMotif,
    seq::AbstractVector{UInt8};
    strands::StrandPolicy=ForwardOnly(),
) where {T<:AbstractFloat}
    n_pos = npositions_ho(length(seq), model)
    if length(dest) < n_pos
        throw(
            ArgumentError("destination has $(length(dest)) elements, need at least $n_pos.")
        )
    end
    return _scan_inplace_ho!(strands, dest, model, seq, n_pos)
end

function _scan_inplace_ho!(
    ::ForwardOnly, dest::AbstractVector{T}, model::AbstractHigherOrderMotif, seq, n_pos
) where {T<:AbstractFloat}
    return scan_forward!(dest, model, seq, n_pos)
end

function _scan_inplace_ho!(
    ::ReverseOnly, dest::AbstractVector{T}, model::AbstractHigherOrderMotif, seq, n_pos
) where {T<:AbstractFloat}
    return scan_reverse!(dest, model, seq, n_pos)
end

function _scan_inplace_ho!(
    ::BestStrand, dest::AbstractVector{T}, model::AbstractHigherOrderMotif, seq, n_pos
) where {T<:AbstractFloat}
    return scan_best!(dest, model, seq, n_pos)
end

function _scan_inplace_ho!(
    ::BothStrands, dest::AbstractVector{T}, model::AbstractHigherOrderMotif, seq, n_pos
) where {T<:AbstractFloat}
    return throw(
        ArgumentError(
            "scan! with BothStrands is not supported; use scan(model, seq; strands=BothStrands()).",
        ),
    )
end

# ── Generic batch scanning (EncodedSequenceBatch) ─────────────────────────

"""
    scan(model::AbstractHigherOrderMotif, batch; strands, execution)

Scan all sequences in a batch with a higher-order model, returning a
[`RaggedArray{Float32}`](@ref) of scores.

For `BothStrands`, returns a [`StrandPair{RaggedArray{Float32}}`](@ref).

Under `ThreadedExecution`, sequences are processed in parallel at the
 top level. Inner scanning kernels remain serial.
Generic method for all AbstractHigherOrderMotif subtypes.
"""
function scan(
    model::AbstractHigherOrderMotif,
    batch::EncodedSequenceBatch;
    strands::StrandPolicy=ForwardOnly(),
    execution::ExecutionPolicy=SerialExecution(),
)
    if strands isa BothStrands
        return _ho_scan_batch_both(
            model,
            batch,
            (sl, m) -> npositions(m, sl),
            (fwd, rev, m, seq, npos) -> scan_both!(fwd, rev, m, seq, npos),
            execution,
        )
    end
    scan_fn! = if strands isa ForwardOnly
        (dest, m, seq, npos) -> scan_forward!(dest, m, seq, npos)
    elseif strands isa ReverseOnly
        (dest, m, seq, npos) -> scan_reverse!(dest, m, seq, npos)
    elseif strands isa BestStrand
        (dest, m, seq, npos) -> scan_best!(dest, m, seq, npos)
    else
        throw(ArgumentError("unsupported strand policy: $(typeof(strands))"))
    end
    return _ho_scan_batch(
        strands, model, batch, (sl, m) -> npositions(m, sl), scan_fn!, execution
    )
end

# ── Generic scan result lengths ─────────────────────────────────────────

"""
    scan_result_lengths(model::AbstractHigherOrderMotif, batch)

Return a `Vector{Int}` with the number of scan positions for each sequence.
Generic method for all AbstractHigherOrderMotif subtypes.
"""
function scan_result_lengths(model::AbstractHigherOrderMotif, batch::EncodedSequenceBatch)
    return [npositions_ho(seqlength(batch, i), model) for i in 1:nsequences(batch)]
end
