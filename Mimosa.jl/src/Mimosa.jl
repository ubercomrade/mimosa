"""
    Mimosa

A Julia package for motif scanning, comparison, and statistical evaluation.

This is Stage 1: PWM/PFM parsing, matrix metrics, motif alignment, and typed
results. See `REFACTORING.md` and `PLAN.md` for the migration roadmap.
"""
module Mimosa

include("errors.jl")
include("models/models.jl")
include("io/io.jl")
include("comparison/comparison.jl")
include("serialization.jl")
include("cli.jl")

export readmodel, read_meme, read_pfm, compare, to_json, to_dict, main
export MimosaError, ModelFormatError, ModelDimensionError, InvariantError

end # module