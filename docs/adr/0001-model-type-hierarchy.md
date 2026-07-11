# ADR 0001: Model Type Hierarchy

## Status

Implemented (Stage 1): `PFM` and `PWM` concrete types implemented in
`Mimosa.jl/src/models/types.jl`. Higher-order model types are planned for
Stage 5.

## Context

The Python implementation uses a single `GenericModel(type_key, name, representation, length, config)` dataclass for all motif model families. The `type_key` string drives dispatch through a `registry: dict[str, ModelHandler]`. The `representation` field is `Any` (NumPy array of varying shape/dtype), and `config` is a `dict` carrying model-specific parameters (`kmer`, `order`, `minimum`, `maximum`, `_source_pfm`, `scores_data`).

This design is idiomatic Python but violates Julia's type stability principles:
- `representation::Any` prevents specialization.
- String-keyed registry prevents multiple dispatch.
- `config::Dict` requires runtime type checks in hot paths.

Six model families must be represented: PWM, PFM, BaMM, SiteGA, Dimont, Slim, plus the pseudo-model ScoreProfile.

## Decision

Implement concrete immutable parametric structs for each model family, sharing an abstract supertype for API-level dispatch only:

```julia
abstract type AbstractMotifModel end
abstract type AbstractMatrixMotif <: AbstractMotifModel end
abstract type AbstractHigherOrderMotif <: AbstractMotifModel end

struct PFM{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractMatrixMotif
    name::String
    frequencies::M  # axes: base × position
end

struct PWM{T<:AbstractFloat,M<:AbstractMatrix{T},B} <: AbstractMatrixMotif
    name::String
    weights::M       # axes: base × position
    background::B    # NTuple{4,T} in the common case
end

struct BaMM{T<:AbstractFloat,A<:AbstractArray{T},I<:Integer} <: AbstractHigherOrderMotif
    name::String
    log_odds::A      # axes: [5, 5, ..., 5, position], ndim = order+2
    order::I
end

struct SiteGA{T<:AbstractFloat,A<:AbstractArray{T}} <: AbstractHigherOrderMotif
    name::String
    weights::A       # axes: 5 × 5 × position
    minimum::T
    maximum::T
end

struct Dimont{T<:AbstractFloat,A<:AbstractArray{T},I<:Integer} <: AbstractHigherOrderMotif
    name::String
    log_odds::A      # axes: [5, ..., 5, position], ndim = span+2
    span::I
end

struct Slim{T<:AbstractFloat,A<:AbstractArray{T},I<:Integer} <: AbstractHigherOrderMotif
    name::String
    log_odds::A      # axes: [5, ..., 5, position], ndim = span+2
    span::I
end

struct ScoreProfile{T<:AbstractFloat}
    name::String
    values::Vector{Vector{T}}  # ragged: one vector per sequence
end
```

### Key choices

1. **PFM and PWM are separate types.** PFM is a frequency matrix (non-negative, per-position distribution). PWM is a log-odds weight matrix. They have different invariants, different constructors, and different uses (PFM for reconstruction output, PWM for scanning). Keeping them separate prevents accidental misuse.

2. **Higher-order models store the 5-ary dense tensor.** The Python implementation materializes Dimont/Slim/BaMM into dense `[5, ..., 5, position]` tensors at load time. Julia will do the same initially, since the scan kernel operates on this layout. A future optimization could store sparse representations, but that requires benchmark evidence.

3. **`background` is parametric (`B`).** Most PWMs use `NTuple{4,T}`, but the type allows for future extensions without changing the struct.

4. **No `length` field.** Motif length is `size(weights, 2)` for matrix motifs and `size(log_odds, ndims)` for higher-order motifs. Storing it separately risks inconsistency.

5. **No `kmer` field.** For BaMM, `kmer = order + 1`. For Dimont/Slim, `kmer = span + 1`. For PWM/SiteGA, `kmer = 1`. This is derivable, not stored.

6. **`ScoreProfile` is not a motif model.** It does not support scanning or site extraction. It participates only in profile comparison. It gets its own type outside the `AbstractMotifModel` hierarchy.

## Alternatives considered

### A. Single `GenericModel` with parametric representation

```julia
struct GenericModel{T,R}
    type_key::Symbol
    name::String
    representation::R
    config::NamedTuple
end
```

Rejected: still requires runtime type inspection to determine behavior. `Symbol` dispatch is no better than string dispatch.

### B. Union type instead of abstract hierarchy

```julia
const MotifModel = Union{PWM, PFM, BaMM, SiteGA, Dimont, Slim}
```

Considered: simpler, no abstract dispatch. But loses the ability to write generic methods for `AbstractMatrixMotif` vs `AbstractHigherOrderMotif`. The hierarchy is shallow (two levels) and each level has practical dispatch uses.

### C. PFM as a view of PWM

Rejected: PFM and PWM have fundamentally different semantics. A PFM can be created from site counts without a background model. A PWM requires a background. Forcing one to contain the other creates unnecessary coupling.

## Consequences

- Adding a new model family requires a new struct + methods. This is by design: it prevents accidental catch-all handling.
- Heterogeneous collections (`Vector{AbstractMotifModel}`) require function barriers or grouping by concrete type for batch operations. This is handled at the API layer, not in hot kernels.
- `ScoreProfile` comparison uses a separate code path from motif comparison, matching Python's `strategy` split.
- Provenance/metadata (source file, tool version, creation timestamp) is NOT in the model struct. It belongs in a separate `ModelProvenance` type attached at I/O boundaries.

## Migration impact

- Python `read_model(path, "pwm")` → Julia `readmodel(path; format=:pwm)` returns a `PWM` or `PFM`.
- Python `model.type_key` → Julia dispatch on concrete type.
- Python `model.config["kmer"]` → Julia: derived from struct fields.
- Python `model.config["_source_pfm"]` → Julia: PWM stores weights only; PFM is a separate object. The `_source_pfm` hack is not needed because `write_pfm` can be called on a `PFM` directly.
- Python `model.config["scores_data"]` → Julia: `ScoreProfile` struct holds the data directly.