# Extending Mimosa

## Adding a new model family

To add a new motif model type to Mimosa.jl, you need:

1. A concrete immutable struct subtyping `AbstractMotifModel`
2. Geometry and score-matrix methods: `motif_length`, `window_size`,
   `scorematrix`, and `is_scannable`
3. A `scorebounds` method and concrete strand kernels
4. A parser registered at the I/O boundary in the existing `readmodel` dispatch
5. Optionally: sites, reconstruction, comparison, writer

## Step 1: Define the type

```julia
struct MyModel{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractMotifModel
    name::String
    representation::M
    # model-specific fields...
end

Base.length(model::MyModel) = size(model.representation, 2)
Base.eltype(::Type{<:MyModel{T}}) where {T} = T
motif_length(model::MyModel) = length(model)
window_size(model::MyModel) = motif_length(model)
scorematrix(model::MyModel) = model.representation
is_scannable(::MyModel) = true
```

## Step 2: Implement score bounds

```julia
function scorebounds(model::MyModel)
    rep = model.representation
    n_cols = size(rep, 2)
    min_score = zero(Float32)
    max_score = zero(Float32)
    for col in 1:n_cols
        col_min = minimum(@view rep[:, col])
        col_max = maximum(@view rep[:, col])
        min_score += col_min
        max_score += col_max
    end
    return (Float32(min_score), Float32(max_score))
end
```

## Step 3: Implement scanning

Implement all four concrete strand kernels: `scan_forward!`, `scan_reverse!`,
`scan_best!`, and `scan_both!`. The public `scan`, `scan!`, and batch methods
then use them after validating encoded input and output geometry:

```julia
function scan_forward!(
    dest::AbstractVector{T}, model::MyModel, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    # Your scanning kernel here
    return dest
end

function scan_reverse!(
    dest::AbstractVector{T}, model::MyModel, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    # Reverse-strand kernel here
    return dest
end
```

For higher-order models, provide the internal geometry traits used by the
shared rolling-k-mer kernels: `kmer`, `context_length`, `scan_width`, and
`site_start_offset`. Keep these implementation details private to the package;
third-party extensions should mirror an existing model family and add focused
serial/threaded equivalence tests.

## Step 4: Add a parser

```julia
function read_mymodel(path::AbstractString; kwargs...)
    # Parse the file, validate, construct the type
    return MyModel(name, representation, ...)
end

# Add the extension to the existing format-detection and dispatch branches in
# `src/io/model_storage.jl`; do not add a competing broad `readmodel` method.
```

## Step 5: Add tests

- Unit tests: constructor invariants, scorebounds, scanning, determinism
- Compatibility tests: compare against reference implementation
- Property tests: round-trip serialization, no mutation, determinism

## Extension principles

- **No central registry**: New models are added via dispatch, not by editing
  a registry dictionary
- **No `Any` fields**: All fields must have concrete or parametric types
- **No string dispatch in hot paths**: String identifiers only at I/O boundary
- **Type stability**: All hot kernel methods must return concrete types
- **No type piracy**: Only define methods for types you own
