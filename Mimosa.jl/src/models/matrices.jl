# Matrix conversion helpers mirroring Python `functions/matrices.py`.

const NUCLEOTIDE_CARDINALITY = 4
const PSEUDOCOUNT_PWM::Float32 = 1e-4

"""
    pcm_to_pfm(pcm; pseudocount=0.25)

Convert a Position Count Matrix to a Position Frequency Matrix.

`pcm` axes: `(base, position)` with `base ∈ 1:4`.
"""
function pcm_to_pfm(
    pcm::AbstractMatrix{T}; pseudocount::AbstractFloat=0.25f0
) where {T<:AbstractFloat}
    if size(pcm, 1) != NUCLEOTIDE_CARDINALITY
        throw(ModelDimensionError("PCM must have 4 rows, got $(size(pcm, 1))."))
    end
    n_sites = sum(pcm; dims=1)
    pc = T(pseudocount)
    denom = n_sites .+ T(4) * pc
    return (pcm .+ pc) ./ denom
end

"""
    pfm_to_pwm(pfm; background=0.25)

Convert a Position Frequency Matrix to a log-odds Position Weight Matrix.

The result has 4 rows (base × position), matching the Python `pfm_to_pwm`
which computes `log((pfm + 0.0001) / 0.25)`.
"""
function pfm_to_pwm(
    pfm::AbstractMatrix{T}; background::AbstractFloat=0.25f0
) where {T<:AbstractFloat}
    if size(pfm, 1) != NUCLEOTIDE_CARDINALITY
        throw(ModelDimensionError("PFM must have 4 rows, got $(size(pfm, 1))."))
    end
    pc = T(PSEUDOCOUNT_PWM)
    bg = T(background)
    return @. log((pfm + pc) / bg)
end

"""
    extend_pwm_with_n(weights4)

Extend a 4-row PWM to a 5-row PWM by appending an N-state row equal to the
per-column minimum, matching Python's `pwm_model_from_pfm`.
"""
function extend_pwm_with_n(weights4::AbstractMatrix{T}) where {T<:AbstractFloat}
    if size(weights4, 1) != NUCLEOTIDE_CARDINALITY
        throw(
            ModelDimensionError(
                "PWM weights must have 4 rows to extend, got $(size(weights4, 1))."
            ),
        )
    end
    n_row = vec(minimum(weights4; dims=1))
    return vcat(weights4, reshape(n_row, 1, :))
end

"""
    pwm_from_pfm(pfm; background=0.25, name="")

Build a ready-to-scan [`PWM`](@ref) from a [`PFM`](@ref) or raw frequency matrix.

This mirrors Python's `pwm_model_from_pfm`: `pfm_to_pwm` then extend with the
per-column minimum as the N-state row.
"""
function pwm_from_pfm(
    pfm::AbstractMatrix{T}; background::AbstractFloat=0.25f0, name::AbstractString=""
) where {T<:AbstractFloat}
    pwm4 = pfm_to_pwm(pfm; background=background)
    weights = extend_pwm_with_n(pwm4)
    bg = ntuple(_ -> T(background), 4)
    return PWM(name, weights, bg)
end

function pwm_from_pfm(
    model::PFM; background::AbstractFloat=0.25f0, name::AbstractString=model.name
)
    return pwm_from_pfm(model.frequencies; background=background, name=name)
end

"""
    reverse_complement(weights)

Return the reverse-complement of a PWM weights matrix.

For the `(base, position)` layout this flips the base rows (A↔T, C↔G) and
reverses the position columns, matching Python's `pwm[::-1, ::-1]`.
"""
function reverse_complement(weights::AbstractMatrix{T}) where {T<:AbstractFloat}
    if size(weights, 1) ∉ (NUCLEOTIDE_CARDINALITY, 5)
        throw(
            ModelDimensionError(
                "reverse_complement expects 4 or 5 rows, got $(size(weights, 1))."
            ),
        )
    end
    return reverse(reverse(weights; dims=1); dims=2)
end

function reverse_complement(model::PWM)
    return PWM(model.name, reverse_complement(model.weights), model.background)
end
reverse_complement(model::PFM) = PFM(model.name, reverse_complement(model.frequencies))

"""
    scorebounds(model::PWM)

Return `(min_score, max_score)` theoretical score bounds for a [`PWM`](@ref).

Mirrors Python's `score_bounds_from_representation`: take the per-column min/max
across all rows and sum across positions.
"""
function scorebounds(model::PWM)
    w = model.weights
    col_min = vec(minimum(w; dims=1))
    col_max = vec(maximum(w; dims=1))
    return (sum(col_min), sum(col_max))
end

function scorebounds(model::PFM)
    w = model.frequencies
    col_min = vec(minimum(w; dims=1))
    col_max = vec(maximum(w; dims=1))
    return (sum(col_min), sum(col_max))
end
