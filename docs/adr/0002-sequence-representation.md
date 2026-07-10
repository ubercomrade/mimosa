# ADR 0002: Sequence Representation

## Status

Proposed (Stage 0)

## Context

The Python implementation uses `SequenceBatch`, a `TypedDict` with a padded int8 2D array (`values: np.ndarray`, `lengths: np.ndarray`, `padding_value: int`). All sequences are padded to the maximum length with value 4 (N/padding). Scanning kernels process the full padded width, using `lengths` to determine valid output positions.

For score profiles and profile bundles, Python uses `MaskedBatch` (float32 2D + boolean mask) and `ProfileBundle` (float32 3D with strand axis).

Problems with padding:
- Wasted computation on padding columns when sequence lengths vary widely.
- Mask arrays double memory usage.
- Bucketing optimization in Python mitigates this but adds complexity.

Julia has better tools for ragged storage and views without copying.

## Decision

Use a flat offset-based ragged representation as canonical storage, with padded views available as kernel scratch buffers:

```julia
struct EncodedSequenceBatch{V<:AbstractVector{UInt8},I<:AbstractVector{Int}}
    data::V      # flat concatenation of all encoded sequences
    offsets::I   # offsets[i]: 1-based start of sequence i; offsets[end] = length(data) + 1
end
```

### Encoding

| Code | Nucleotide(s) |
|---|---|
| 0x00 | A |
| 0x01 | C |
| 0x02 | G |
| 0x03 | T |
| 0x04 | N / padding / ambiguous |

Lowercase is normalized to uppercase before encoding. All non-ACGT characters (including IUPAC codes) map to 4, matching Python behavior exactly. This is a deliberate simplification, not a bug to fix.

### Constructor invariants

- `offsets[1] == 1` (Julia one-based indexing).
- `offsets` is monotonically non-decreasing.
- `offsets[end] == length(data) + 1`.
- Empty sequences are allowed: consecutive equal offsets.
- Number of sequences: `length(offsets) - 1`.

### Access patterns

```julia
sequence(batch, i) = @view batch.data[batch.offsets[i]:(batch.offsets[i+1]-1)]
length(batch, i) = batch.offsets[i+1] - batch.offsets[i]
nsequences(batch) = length(batch.offsets) - 1
```

### Score profiles

For score profiles (precomputed or scanned), use a separate ragged type:

```julia
struct RaggedArray{T,V<:AbstractVector{T},I<:AbstractVector{Int}}
    data::V
    offsets::I
end
```

Same invariants as `EncodedSequenceBatch` but parametric in element type. Used for:
- Single-strand score tracks: `RaggedArray{Float32}`.
- Score profile input (ScoreProfile pseudo-model).

For two-strand profile bundles, use a struct with two `RaggedArray` fields (forward/reverse) rather than a 3D padded array:

```julia
struct StrandProfileBundle{T}
    forward::RaggedArray{T}
    reverse::RaggedArray{T}
    offsets::Vector{Int}  # shared, since both strands have same lengths
end
```

### Padded scratch buffers

When a kernel benefits from contiguous dense access (e.g., Numba-style bucket processing), the API can provide a function to copy ragged data into a padded `Matrix` with explicit lengths:

```julia
function padded_view(batch::EncodedSequenceBatch)
    # returns (Matrix{UInt8}, Vector{Int}, padding_value) for scratch use
end
```

This is NOT the canonical representation. It is an implementation detail of specific kernels, decided by benchmark.

## Alternatives considered

### A. `Vector{Vector{UInt8}}` (array of arrays)

Rejected: no contiguous memory layout, poor cache locality for batch scanning, no O(1) total-size query.

### B. `BioSequences.LongDNA` as core type

Considered as an interop layer. `LongDNA` stores 2-bit encoded sequences, which is more compact. But:
- 2-bit encoding cannot represent N/ambiguous without a separate mask.
- The scan kernel needs 5-ary encoding for context codes; converting from 2-bit adds overhead in the inner loop.
- `BioSequences` is a heavy dependency for a feature that may not be needed.

Decision: evaluate `BioSequences` as an optional extension for FASTA I/O interop, but not as core representation. The flat `UInt8` buffer with offsets is simpler and sufficient.

### C. Keep padded dense as canonical (match Python)

Rejected: padding wastes memory and computation when lengths vary. The mask array is an artifact of dense storage, not a domain concept. Ragged storage is the natural representation for variable-length sequences.

### D. Bit-packed 5-ary encoding (3 bits per base)

Considered for memory compactness. 5 states fit in 3 bits vs 8 bits for `UInt8`. But:
- Byte-aligned access is simpler and faster on modern hardware.
- 3-bit packing complicates indexing and vectorization.
- Memory savings (37%) are not a bottleneck for the fixture sizes in the plan.

Decision: use `UInt8` initially. Revisit if profiling shows memory bandwidth is the bottleneck.

## Consequences

- `EncodedSequenceBatch` is immutable; operations create new batches.
- Views (`@view batch.data[off:off+len-1]`) provide zero-copy sequence access.
- Scanning kernels iterate over ragged rows, not padded matrices. Each row's length determines the number of output positions.
- For batch-parallel kernels, a padded scratch buffer may be created, but this is a kernel implementation detail, not part of the public API.
- `BioSequences` integration is deferred to an optional extension.

## Migration impact

- Python `SequenceBatch` (dict with `values`, `lengths`, `padding_value`) → Julia `EncodedSequenceBatch` (struct with `data`, `offsets`).
- Python `batch["values"][i, :lengths[i]]` → Julia `sequence(batch, i)`.
- Python `batch["lengths"][i]` → Julia `length(batch, i)`.
- Python `make_random_sequence_batch(n, len, seed)` → Julia function using `AbstractRNG`, generating the same logical sequences (but NOT the same random bytes as NumPy — fixtures store encoded bytes).