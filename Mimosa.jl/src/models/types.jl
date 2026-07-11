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

"""
    PWM{T,M,B}

Position Weight Matrix: log-odds weights for scanning.

`weights` uses axes `(base, position)` with `base ∈ 1:5` (A, C, G, T, N).
The fifth row holds the N-state score (minimum over concrete bases), matching
the Python representation that materializes a 5-row extended PWM.
`background` is a 4-tuple of nucleotide background frequencies.
"""
struct PWM{T<:AbstractFloat,M<:AbstractMatrix{T},B<:NTuple{4,AbstractFloat}} <: AbstractMatrixMotif
    name::String
    weights::M
    background::B
end

function PWM(name::AbstractString, weights::AbstractMatrix{T}, background::NTuple{4}) where {T<:AbstractFloat}
    _validate_pwm_weights(weights)
    PWM{T,typeof(weights),typeof(background)}(String(name), weights, background)
end

function _validate_pwm_weights(weights::AbstractMatrix)
    if size(weights, 1) != 5
        throw(ModelDimensionError("PWM weights must have 5 rows (A,C,G,T,N), got $(size(weights, 1))."))
    end
    if size(weights, 2) < 1
        throw(ModelDimensionError("PWM motif length must be positive, got $(size(weights, 2))."))
    end
    if !all(isfinite, weights)
        throw(ModelFormatError("", "PWM weights contain non-finite values."))
    end
    nothing
end

Base.length(model::PFM) = size(model.frequencies, 2)
Base.length(model::PWM) = size(model.weights, 2)

Base.eltype(::Type{<:PFM{T}}) where {T} = T
Base.eltype(::Type{<:PWM{T}}) where {T} = T

Base.size(model::PFM) = size(model.frequencies)
Base.size(model::PWM) = size(model.weights)

Base.show(io::IO, model::PFM) = print(io, "PFM(\"$(model.name)\", $(size(model.frequencies)))")
Base.show(io::IO, model::PWM) = print(io, "PWM(\"$(model.name)\", $(size(model.weights)))")