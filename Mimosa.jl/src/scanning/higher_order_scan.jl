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
    seq_len = length(seq)
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
    seq_len = length(seq)
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
    seq_len = length(seq)
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
    seq_len = length(seq)
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
