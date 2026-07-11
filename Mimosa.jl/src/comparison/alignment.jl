# Direct motif matrix alignment: sliding offset over all four orientations.

# reverse_complement and prepare_motif helpers are defined in the enclosing
# Mimosa module scope via include order.

# Orientation definitions per ADR 0006.
"""
    Orientation

One of the four orientation candidates for motif alignment: `++`, `+-`, `-+`,
`--`. `rank` encodes the tie-break priority (lower wins on equal score).
"""
struct Orientation
    label::String
    rank::Int
end

const ORIENTATIONS = (
    Orientation("++", 0), Orientation("+-", 1), Orientation("-+", 2), Orientation("--", 3)
)

function orientation_rank(label::AbstractString)
    for o in ORIENTATIONS
        o.label == label && return o.rank
    end
    return throw(ArgumentError("invalid orientation label: $label."))
end

"""
    MotifCandidate

One scored orientation/offset candidate for motif matrix alignment.
"""
struct MotifCandidate
    orientation::Orientation
    offset::Int
    score::Float32
end

"""
    align_motif_matrices(query_matrix, target_matrix, metric)

Slide `query_matrix` over `target_matrix` across all offsets and return the
best `(score, offset)` for one orientation pair. Mirrors Python's
`_align_motif_matrices` including the minimum-overlap policy and the
negative-to-positive offset traversal order with first-wins tie-breaking.
"""
function align_motif_matrices(
    query::AbstractMatrix{T}, target::AbstractMatrix{T}, metric::AbstractColumnMetric
) where {T<:AbstractFloat}
    query_length = size(query, 2)
    target_length = size(target, 2)
    min_overlap = min(query_length, target_length) / 2.0
    best_score = T(-Inf)
    best_offset = 0
    for offset in (-(target_length - 1)):(query_length - 1)
        if offset < 0
            overlap = min(query_length, target_length + offset)
            if Float64(overlap) < min_overlap
                continue
            end
            qcols = view(query, :, 1:overlap)
            tcols = view(target, :, (1 - offset):((1 - offset) + overlap - 1))
        else
            overlap = min(query_length - offset, target_length)
            if Float64(overlap) < min_overlap
                continue
            end
            qcols = view(query, :, (offset + 1):(offset + overlap))
            tcols = view(target, :, 1:overlap)
        end
        score = score_columns(metric, qcols, tcols)
        if score > best_score
            best_score = score
            best_offset = offset
        end
    end
    return (Float32(best_score), best_offset)
end

"""
    score_motif_candidates(query_forward, query_reverse, target_forward, target_reverse, metric)

Score all four orientation candidates for one prepared motif pair, returning a
vector of [`MotifCandidate`](@ref) in the canonical evaluation order
(`++`, `+-`, `-+`, `--`).
"""
function score_motif_candidates(
    q_fwd::AbstractMatrix{T},
    q_rev::AbstractMatrix{T},
    t_fwd::AbstractMatrix{T},
    t_rev::AbstractMatrix{T},
    metric::AbstractColumnMetric,
) where {T<:AbstractFloat}
    candidates = Vector{MotifCandidate}(undef, 4)
    pairs = (
        (ORIENTATIONS[1], q_fwd, t_fwd),
        (ORIENTATIONS[2], q_fwd, t_rev),
        (ORIENTATIONS[3], q_rev, t_fwd),
        (ORIENTATIONS[4], q_rev, t_rev),
    )
    for (i, (orient, q, t)) in enumerate(pairs)
        score, offset = align_motif_matrices(q, t, metric)
        candidates[i] = MotifCandidate(orient, offset, score)
    end
    return candidates
end

"""
    select_best(candidates)

Select the best candidate with deterministic tie-breaking per ADR 0006:
higher score wins; on equal score, lower orientation rank wins; on equal rank,
earlier evaluation order wins.
"""
function select_best(candidates::AbstractVector{MotifCandidate})
    best = candidates[1]
    for c in candidates[2:end]
        if c.score > best.score
            best = c
        elseif c.score == best.score && c.orientation.rank < best.orientation.rank
            best = c
        end
    end
    return best
end

"""
    prepare_motif(weights)

Return `(forward, reverse)` flattened views of a PWM for alignment.
The forward view is the 4-row core (A,C,G,T); the reverse view is the
reverse complement of the 4-row core.
"""
function prepare_motif(weights::AbstractMatrix{T}) where {T<:AbstractFloat}
    if size(weights, 1) == 5
        forward = view(weights, 1:4, :)
    else
        forward = weights
    end
    rc = reverse_complement(forward)
    return forward, rc
end
