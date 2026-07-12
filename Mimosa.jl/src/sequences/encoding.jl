# Encoded sequence batch: flat UInt8 buffer with offsets for ragged DNA sequences.
# Per ADR 0002: A=0x00, C=0x01, G=0x02, T=0x03, N/ambiguous/padding=0x04.

const N_CODE = 0x04

# 256-entry lookup table: maps ASCII byte to 5-ary nucleotide code.
# Indexed by byte+1 (Julia 1-based), so _ENCODE_TABLE[byte + 1] gives the code.
const _ENCODE_TABLE = fill(N_CODE, 256)
_ENCODE_TABLE[Int('A') + 1] = 0x00
_ENCODE_TABLE[Int('C') + 1] = 0x01
_ENCODE_TABLE[Int('G') + 1] = 0x02
_ENCODE_TABLE[Int('T') + 1] = 0x03
_ENCODE_TABLE[Int('a') + 1] = 0x00
_ENCODE_TABLE[Int('c') + 1] = 0x01
_ENCODE_TABLE[Int('g') + 1] = 0x02
_ENCODE_TABLE[Int('t') + 1] = 0x03

"""
    encode_base(byte::UInt8)

Encode one ASCII byte to a 5-ary nucleotide code.
A=0, C=1, G=2, T=3, all others (including N and IUPAC) = 4.
"""
encode_base(byte::UInt8) = _ENCODE_TABLE[byte + 1]

"""
    encode_sequence(str::AbstractString)

Encode a DNA string to a `Vector{UInt8}` using 5-ary encoding.
"""
function encode_sequence(str::AbstractString)
    n = ncodeunits(str)
    result = Vector{UInt8}(undef, n)
    @inbounds for i in 1:n
        result[i] = _ENCODE_TABLE[codeunit(str, i) + 1]
    end
    return result
end

"""
    EncodedSequenceBatch{V,I}

A batch of DNA sequences stored as a flat `UInt8` buffer with offsets.

# Invariants
- `offsets[1] == 1`.
- `offsets` is monotonically non-decreasing.
- `offsets[end] == length(data) + 1`.
- Encoding: A=0x00, C=0x01, G=0x02, T=0x03, N/ambiguous=0x04.

See ADR 0002 for the design rationale.
"""
struct EncodedSequenceBatch{V<:AbstractVector{UInt8},I<:AbstractVector{Int}}
    data::V
    offsets::I

    function EncodedSequenceBatch{V,I}(
        data::V, offsets::I
    ) where {V<:AbstractVector{UInt8},I<:AbstractVector{Int}}
        _validate_ragged_offsets(offsets, length(data))
        return new{V,I}(data, offsets)
    end
end

function EncodedSequenceBatch(data::AbstractVector{UInt8}, offsets::AbstractVector{Int})
    return EncodedSequenceBatch{typeof(data),typeof(offsets)}(data, offsets)
end

"""
    nsequences(batch::EncodedSequenceBatch)

Return the number of sequences in a batch.
"""
nsequences(batch::EncodedSequenceBatch) = length(batch.offsets) - 1

"""
    seqlength(batch::EncodedSequenceBatch, i::Int)

Return the length of sequence `i` in a batch.
"""
seqlength(batch::EncodedSequenceBatch, i::Int) = batch.offsets[i + 1] - batch.offsets[i]

"""
    sequence(batch::EncodedSequenceBatch, i::Int)

Return a zero-copy view of sequence `i` in a batch.
"""
function sequence(batch::EncodedSequenceBatch, i::Int)
    len = seqlength(batch, i)
    if len == 0
        # Return a safe empty view at valid indices
        return view(batch.data, 1:0)
    end
    return @view batch.data[batch.offsets[i]:(batch.offsets[i + 1] - 1)]
end

Base.length(batch::EncodedSequenceBatch) = nsequences(batch)
Base.IteratorSize(::Type{<:EncodedSequenceBatch}) = Base.HasLength()
Base.IteratorEltype(::Type{<:EncodedSequenceBatch}) = Base.EltypeUnknown()

function Base.iterate(batch::EncodedSequenceBatch, state::Int=1)
    state > nsequences(batch) && return nothing
    return (sequence(batch, state), state + 1)
end

function Base.:(==)(a::EncodedSequenceBatch, b::EncodedSequenceBatch)
    return a.offsets == b.offsets && a.data == b.data
end

function Base.show(io::IO, batch::EncodedSequenceBatch)
    return print(
        io,
        "EncodedSequenceBatch($(nsequences(batch)) sequences, $(length(batch.data)) bytes)",
    )
end

"""
    empty_sequence_batch()

Return an empty [`EncodedSequenceBatch`](@ref) with zero sequences.
"""
empty_sequence_batch() = EncodedSequenceBatch(UInt8[], [1])

# ── Random sequence generation ──────────────────────────────────────────────

# Lookup from a uniform Float64 in [0, 1) to a nucleotide code.
# Bases are assigned equal-width intervals: A=[0,0.25), C=[0.25,0.5),
# G=[0.5,0.75), T=[0.75,1.0).
const _BASE_LOOKUP = (0x00, 0x01, 0x02, 0x03)

"""
    make_random_sequences(n::Int, len::Int; seed::Integer=127)

Generate `n` random DNA sequences of length `len` each, using a seeded
`MersenneTwister` RNG. Bases are drawn uniformly from A, C, G, T.

Returns an [`EncodedSequenceBatch`](@ref). Reproducible within Julia but
not bit-compatible with Python's `np.random.default_rng` (different RNG
algorithm). This is acceptable for CLI fallback sequences; users should
provide explicit FASTA for scientific reproducibility across languages.
"""
function make_random_sequences(n::Int, len::Int; seed::Integer=127)
    n < 0 && throw(ArgumentError("n must be non-negative, got $n."))
    len < 0 && throw(ArgumentError("len must be non-negative, got $len."))
    rng = Random.MersenneTwister(seed)
    rows = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        row = Vector{UInt8}(undef, len)
        for j in 1:len
            idx = floor(Int, rand(rng) * 4.0) + 1
            row[j] = _BASE_LOOKUP[idx]
        end
        rows[i] = row
    end
    return EncodedSequenceBatch(rows)
end

"""
    EncodedSequenceBatch(rows::AbstractVector{<:AbstractVector{UInt8}})

Build an [`EncodedSequenceBatch`](@ref) from a vector of encoded sequence vectors.
"""
function EncodedSequenceBatch(rows::AbstractVector{<:AbstractVector{UInt8}})
    n = length(rows)
    offsets = Vector{Int}(undef, n + 1)
    offsets[1] = 1
    for i in 1:n
        offsets[i + 1] = offsets[i] + length(rows[i])
    end
    data = Vector{UInt8}(undef, offsets[end] - 1)
    for i in 1:n
        r = rows[i]
        dest_start = offsets[i]
        for j in eachindex(r)
            data[dest_start + j - 1] = r[j]
        end
    end
    return EncodedSequenceBatch(data, offsets)
end

"""
    reverse_complement(seq::AbstractVector{UInt8})

Return the reverse complement of an encoded DNA sequence.

For 5-ary encoding: complement of A(0)↔T(3), C(1)↔G(2), N(4) stays N(4).
The result is a new vector; the input is not modified.
"""
function reverse_complement(seq::AbstractVector{UInt8})
    n = length(seq)
    result = Vector{UInt8}(undef, n)
    @inbounds for i in 1:n
        b = seq[n - i + 1]
        result[i] = b == N_CODE ? N_CODE : 0x03 - b
    end
    return result
end

"""
    reverse_complement!(dest::AbstractVector{UInt8}, src::AbstractVector{UInt8})

Write the reverse complement of `src` into `dest`. Both must have the same length.
"""
function reverse_complement!(dest::AbstractVector{UInt8}, src::AbstractVector{UInt8})
    n = length(src)
    length(dest) >= n || throw(ArgumentError("dest must be at least as long as src."))
    @inbounds for i in 1:n
        b = src[n - i + 1]
        dest[i] = b == N_CODE ? N_CODE : 0x03 - b
    end
    return dest
end

"""
    reverse_complement(batch::EncodedSequenceBatch)

Return a new [`EncodedSequenceBatch`](@ref) where every sequence is reverse-complemented.
"""
function reverse_complement(batch::EncodedSequenceBatch)
    n = nsequences(batch)
    rows = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        rows[i] = reverse_complement(sequence(batch, i))
    end
    return EncodedSequenceBatch(rows)
end

# Padded conversion helpers for compatibility with oracle fixtures and
# kernel scratch buffers.

"""
    to_padded(batch::EncodedSequenceBatch; padding::UInt8=N_CODE)

Return `(matrix, lengths)` where `matrix` is a dense padded `Matrix{UInt8}`
with `padding` filling unused columns, and `lengths` is the per-sequence length vector.

This is NOT the canonical representation; it is provided for compatibility
testing and kernel scratch buffers.
"""
function to_padded(batch::EncodedSequenceBatch; padding::UInt8=N_CODE)
    n = nsequences(batch)
    n == 0 && return (Matrix{UInt8}(undef, 0, 0), Int[])
    max_len = maximum(i -> seqlength(batch, i), 1:n)
    matrix = fill(padding, n, max_len)
    lengths = Vector{Int}(undef, n)
    for i in 1:n
        len = seqlength(batch, i)
        lengths[i] = len
        for j in 1:len
            matrix[i, j] = batch.data[batch.offsets[i] + j - 1]
        end
    end
    return (matrix, lengths)
end

"""
    from_padded(values::AbstractMatrix{UInt8}, lengths::AbstractVector{Int}; padding::UInt8=N_CODE)

Build an [`EncodedSequenceBatch`](@ref) from a padded dense matrix and a lengths vector.
Only the first `lengths[i]` columns of each row are used.
"""
function from_padded(
    values::AbstractMatrix{UInt8}, lengths::AbstractVector{Int}; padding::UInt8=N_CODE
)
    n = length(lengths)
    rows = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        len = lengths[i]
        rows[i] = collect(@view values[i, 1:len])
    end
    return EncodedSequenceBatch(rows)
end
