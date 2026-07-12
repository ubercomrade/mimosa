# Extending Mimosa

## Adding a new model family

To add a new motif model type to Mimosa.jl, you need:

1. A concrete immutable struct subtyping `AbstractMotifModel`
2. A `scorebounds` method
3. A `scan` / `scan!` method
4. A parser (via `readmodel` dispatch)
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

For matrix models (similar to PWM):

```julia
function scan_forward!(
    dest::AbstractVector{T}, model::MyModel, seq::AbstractVector{UInt8}, n_pos::Int
) where {T<:AbstractFloat}
    # Your scanning kernel here
    return dest
end

function scan(model::MyModel, seq::AbstractVector{UInt8}; strands::StrandPolicy=ForwardOnly())
    # Dispatch to scan_forward!, scan_reverse!, etc.
end
```

For higher-order models, you can reuse the generic `_ho_scan_forward!` /
`_ho_scan_reverse!` kernels by providing the correct geometry (kmer, context,
window, n_terms).

## Step 4: Add a parser

```julia
function read_mymodel(path::AbstractString; kwargs...)
    # Parse the file, validate, construct the type
    return MyModel(name, representation, ...)
end

# Register with auto-detection
function readmodel(path::AbstractString; format=:auto, kwargs...)
    if format == :auto && endswith(path, ".mymodel")
        return read_mymodel(path; kwargs...)
    end
    # ... existing format dispatch
end
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