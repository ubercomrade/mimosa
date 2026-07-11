# Concrete motif model types per ADR 0001.

"""
    AbstractMotifModel

Abstract supertype of all motif model families (PWM, PFM, BaMM, SiteGA, etc.).
"""
abstract type AbstractMotifModel end

"""
    AbstractMatrixMotif

Abstract supertype of matrix-based motif models (`PFM`, `PWM`).
"""
abstract type AbstractMatrixMotif <: AbstractMotifModel end

"""
    AbstractHigherOrderMotif

Abstract supertype of higher-order motif models (BaMM, SiteGA, Dimont, Slim).
"""
abstract type AbstractHigherOrderMotif <: AbstractMotifModel end

"""
    PFM{T,M}

Position Frequency Matrix: non-negative per-position nucleotide frequencies.

`frequencies` uses axes `(base, position)` with `base ∈ 1:4` (A, C, G, T).
"""
struct PFM{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractMatrixMotif
    name::String
    frequencies::M
end

function PFM(name::AbstractString, frequencies::AbstractMatrix{T}) where {T<:AbstractFloat}
    return PFM{T,typeof(frequencies)}(String(name), frequencies)
end

"""
    PWM{T,M,B}

Position Weight Matrix: log-odds weights for scanning.

`weights` uses axes `(base, position)` with `base ∈ 1:5` (A, C, G, T, N).
The fifth row holds the N-state score (minimum over concrete bases), matching
the Python representation that materializes a 5-row extended PWM.
`background` is a 4-tuple of nucleotide background frequencies.
"""
struct PWM{T<:AbstractFloat,M<:AbstractMatrix{T},B<:NTuple{4,AbstractFloat}} <:
       AbstractMatrixMotif
    name::String
    weights::M
    background::B

    function PWM{T,M,B}(
        name::String, weights::M, background::B
    ) where {T<:AbstractFloat,M<:AbstractMatrix{T},B<:NTuple{4,AbstractFloat}}
        _validate_pwm_weights(weights)
        return new{T,M,B}(name, weights, background)
    end
end

function PWM(
    name::AbstractString, weights::AbstractMatrix{T}, background::NTuple{4}
) where {T<:AbstractFloat}
    return PWM{T,typeof(weights),typeof(background)}(String(name), weights, background)
end

function _validate_pwm_weights(weights::AbstractMatrix)
    if size(weights, 1) != 5
        throw(
            ModelDimensionError(
                "PWM weights must have 5 rows (A,C,G,T,N), got $(size(weights, 1))."
            ),
        )
    end
    if size(weights, 2) < 1
        throw(
            ModelDimensionError(
                "PWM motif length must be positive, got $(size(weights, 2))."
            ),
        )
    end
    if !all(isfinite, weights)
        throw(ModelFormatError("", "PWM weights contain non-finite values."))
    end
    return nothing
end

Base.length(model::PFM) = size(model.frequencies, 2)
Base.length(model::PWM) = size(model.weights, 2)

Base.eltype(::Type{<:PFM{T}}) where {T} = T
Base.eltype(::Type{<:PWM{T}}) where {T} = T

Base.size(model::PFM) = size(model.frequencies)
Base.size(model::PWM) = size(model.weights)

function Base.show(io::IO, model::PFM)
    return print(io, "PFM(\"$(model.name)\", $(size(model.frequencies)))")
end
Base.show(io::IO, model::PWM) = print(io, "PWM(\"$(model.name)\", $(size(model.weights)))")

Base.:(==)(a::PFM, b::PFM) = a.name == b.name && a.frequencies == b.frequencies
function Base.:(==)(a::PWM, b::PWM)
    return a.name == b.name && a.weights == b.weights && a.background == b.background
end

function Base.isapprox(a::PFM, b::PFM; kwargs...)
    return a.name == b.name && isapprox(a.frequencies, b.frequencies; kwargs...)
end

function Base.isapprox(a::PWM, b::PWM; kwargs...)
    return a.name == b.name &&
           isapprox(a.weights, b.weights; kwargs...) &&
           isapprox(collect(a.background), collect(b.background); kwargs...)
end
