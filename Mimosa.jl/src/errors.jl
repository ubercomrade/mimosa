# Small, meaningful exception hierarchy for Mimosa.

"""
    MimosaError

Abstract supertype of all Mimosa-specific exceptions.
"""
abstract type MimosaError <: Exception end

"""
    ModelFormatError

Raised when a model file is malformed, has an unsupported format, or contains
invalid data. `path` is the file path (may be empty for in-memory errors).
"""
struct ModelFormatError <: MimosaError
    path::String
    message::String
end

"""
    ModelDimensionError

Raised when a model has incompatible or invalid dimensions.
"""
struct ModelDimensionError <: MimosaError
    message::String
end

"""
    InvariantError

Raised when an internal invariant is violated.
"""
struct InvariantError <: MimosaError
    message::String
end

function Base.showerror(io::IO, e::ModelFormatError)
    print(io, "ModelFormatError: $(e.path): $(e.message)")
end

function Base.showerror(io::IO, e::ModelDimensionError)
    print(io, "ModelDimensionError: $(e.message)")
end

function Base.showerror(io::IO, e::InvariantError)
    print(io, "InvariantError: $(e.message)")
end