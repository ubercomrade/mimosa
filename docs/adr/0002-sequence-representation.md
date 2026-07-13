# ADR 0002: Sequence and Ragged Representation

## Status

Accepted and implemented.

## Decision

Use flat offset-based storage as the canonical representation for sequences and
score tracks:

```julia
struct EncodedSequenceBatch{V<:AbstractVector{UInt8},I<:AbstractVector{Int}}
    data::V
    offsets::I
end

struct RaggedArray{T,V<:AbstractVector{T},I<:AbstractVector{Int}}
    data::V
    offsets::I
end
```

Offsets are one-based, monotonic, start at 1, and end at `length(data) + 1`.
Equal adjacent offsets preserve empty rows. Row order is exact and stable.

DNA encoding is:

| Base | Code |
|---|---|
| A | `0x00` |
| C | `0x01` |
| G | `0x02` |
| T | `0x03` |
| N or ambiguous | `0x04` |

Public `EncodedSequenceBatch` constructors validate offsets and every encoded
byte. Only audited internal construction paths may use the unsafe constructor
token for already proven data.

Single-strand scan output is `RaggedArray{Float32}`. Two-strand output is a
`StrandPair` containing forward and reverse ragged arrays with matching row
geometry. Padded matrices are explicit conversion outputs or temporary scratch,
never canonical storage.

## Consequences

- Ragged batches avoid padding work and preserve empty/unequal-length rows.
- Hot batch paths should allocate final flat output directly and avoid
  `Vector{Vector}` staging.
- Public row access returns views into contiguous storage.
- Constructors and low-level scan boundaries require one-based axes where the
  kernel relies on them.
- `to_padded` and `from_padded` are interoperability helpers, not alternate
  runtime representations.
