# Sequence representation, encoding, FASTA I/O, and ragged arrays for Mimosa.

using Random

include("ragged.jl")
include("encoding.jl")
include("fasta.jl")

export RaggedArray,
    nrows,
    rowlength,
    row,
    build_ragged,
    empty_ragged,
    EncodedSequenceBatch,
    nsequences,
    seqlength,
    sequence,
    empty_sequence_batch,
    make_random_sequences,
    encode_base,
    encode_sequence,
    reverse_complement,
    reverse_complement!,
    to_padded,
    from_padded,
    read_fasta,
    readsequences,
    N_CODE
