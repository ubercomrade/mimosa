"""
    Mimosa

A Julia package for motif scanning, comparison, and statistical evaluation.

Stage 1: PWM/PFM parsing, matrix metrics, motif alignment, and typed results.
Stage 2: Encoded sequence batches, FASTA reader, PWM scanning with strand
policies.

See `REFACTORING.md` and `PLAN.md` for the migration roadmap.
"""
module Mimosa

include("errors.jl")
include("models/models.jl")
include("sequences/sequences.jl")
include("scanning/scanning.jl")
include("io/io.jl")
include("comparison/comparison.jl")
include("serialization.jl")
include("cli.jl")

export readmodel, read_meme, read_pfm, read_fasta, compare, to_json, to_dict, main
export MimosaError, ModelFormatError, ModelDimensionError, InvariantError

# Sequence / scanning exports
export EncodedSequenceBatch,
    nsequences,
    seqlength,
    sequence,
    empty_sequence_batch,
    encode_base,
    encode_sequence,
    reverse_complement,
    reverse_complement!,
    to_padded,
    from_padded,
    N_CODE
export RaggedArray, nrows, rowlength, row, build_ragged, empty_ragged
export StrandPolicy, ForwardOnly, ReverseOnly, BestStrand, BothStrands, StrandPair
export scan,
    scan!,
    scan_forward!,
    scan_reverse!,
    scan_best!,
    scan_both!,
    npositions,
    scan_result_lengths

end # module
