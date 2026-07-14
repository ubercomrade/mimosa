# ScoreProfile: pseudo-model for precomputed score profiles.

"""
    ScoreProfile

A pseudo-model holding precomputed per-position score profiles read from
FASTA-like numerical files. Both strands resolve to the same scores.

Fields:
- `name::String`: model name (derived from filename).
- `scores::RaggedArray{Float32}`: one row per sequence, variable-length.
"""
struct ScoreProfile <: AbstractMotifModel
    name::String
    scores::RaggedArray{Float32}
end

is_precomputed_profile(::ScoreProfile) = true
motif_length(::ScoreProfile) = throw(ArgumentError("ScoreProfile has no motif length."))
window_size(::ScoreProfile) = throw(ArgumentError("ScoreProfile has no motif window."))

Base.length(model::ScoreProfile) = nrows(model.scores)
function Base.show(io::IO, model::ScoreProfile)
    return print(io, "ScoreProfile(\"$(model.name)\", $(nrows(model.scores)) rows)")
end

function Base.:(==)(a::ScoreProfile, b::ScoreProfile)
    return a.name == b.name && a.scores == b.scores
end

"""
    scorebounds(model::ScoreProfile)

Return `(min_score, max_score)` from the precomputed score values.
"""
function scorebounds(model::ScoreProfile)
    if isempty(model.scores.data)
        return (0.0f0, 0.0f0)
    end
    return (minimum(model.scores.data), maximum(model.scores.data))
end

"""
    profile_bundle(model::ScoreProfile)

Return a `StrandPair{RaggedArray{Float32}}` where both strands are the
same precomputed scores. Matches Python's `_scan_scores_both`.
"""
function profile_bundle(model::ScoreProfile)
    return StrandPair(model.scores, model.scores)
end
