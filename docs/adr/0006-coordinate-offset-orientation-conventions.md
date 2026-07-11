# ADR 0006: Coordinate, Offset, and Orientation Conventions

## Status

Implemented (Stage 1): offset traversal, orientation tie-breaking, and
reverse-complement scoring semantics are implemented in
`Mimosa.jl/src/comparison/alignment.jl` and `Mimosa.jl/src/models/matrices.jl`.

## Context

The Python implementation uses zero-based indexing internally (NumPy convention) but presents zero-based half-open coordinates in CLI output. Offsets in motif alignment can be negative. Orientations are labeled `++`, `+-`, `-+`, `--`. The tie-breaking order and offset traversal direction affect which result is chosen when scores are equal.

These conventions must be frozen before implementing Julia, because:
1. Julia uses one-based indexing internally.
2. CLI output must maintain backward compatibility with Python users.
3. Tie-breaking rules must be exact-matched or documented as divergences.
4. Reverse-complement coordinate correspondence is non-obvious and error-prone.

## Decision

### 6.1 Internal indexing

Julia uses **one-based inclusive** indexing internally for all coordinate computations. This is natural Julia and does not affect external output. No attempt is made to preserve Python's zero-based internal indexing.

### 6.2 CLI output coordinates

CLI JSON output preserves **Python-style zero-based half-open** coordinates for `start` and `end` fields, matching the existing Python output:

```json
{
  "start": 42,
  "end": 54,
  "orientation": "++",
  "offset": -3
}
```

- `start`: zero-based position of the first nucleotide of the motif in the sequence.
- `end`: zero-based position one past the last nucleotide (`end = start + motif_length`).
- Conversion from internal one-based to external zero-based happens in the serializer only.

### 6.3 Offset semantics

**Motif alignment offset**: the offset of the query relative to the target. A positive offset means the query is shifted to the right; a negative offset means the query is shifted to the left.

Python iterates offsets from `-(target_length - 1)` to `query_length - 1` (negative to positive). When scores are equal, the **first** offset in this traversal order wins (strictly-greater comparison, `>` not `>=`).

Julia must preserve this traversal order and tie-breaking for exact compatibility. The offset value reported in the result is the same integer, independent of indexing convention.

### 6.4 Profile alignment shift

The profile shift is the displacement between query and target anchor positions. Range: `[-search_range, search_range]`. Python iterates from `-search_range` to `+search_range`.

Tie-breaking for profile shifts:
1. Higher score wins.
2. If equal: more sites wins (`n_sites`).
3. If equal: smaller `|shift|` wins.
4. If still equal: earlier shift in iteration order (from negative to positive).

### 6.5 Orientation conventions

Four orientation candidates, representing the combination of query and target strand:

| Label | Query strand | Target strand | Tie-break rank |
|---|---|---|---|
| `++` | forward | forward | 0 (highest priority) |
| `+-` | forward | reverse | 1 |
| `-+` | reverse | forward | 2 |
| `--` | reverse | reverse | 3 (lowest priority) |

Tie-breaking: `max(score, -rank)`. When scores are equal, the orientation with the lower rank wins.

Julia must preserve:
- The label strings (`++`, `+-`, `-+`, `--`) in CLI output.
- The tie-breaking order.
- The candidate evaluation order (not parallel-dependent).

### 6.6 Reverse-complement scoring

For a sequence of length `L` and a motif of width `W`:
- Forward score at position `p` (zero-based, `0 ≤ p ≤ L - W`): scores the window `seq[p:p+W]` against the forward motif.
- Reverse score at position `p`: scores the reverse complement of the same window `seq[p:p+W]` against the forward motif. The reverse complement is NOT of a different window — it is the same physical window, read in reverse order with complemented bases.

The reverse score track at position `p` corresponds to scanning the reverse complement of the entire sequence at position `L - p - W` (zero-based). This means:
- Forward track position `p` ↔ reverse track position `L - p - W`.
- The "best" strand at position `p` takes `max(forward[p], reverse[p])`.

Julia implementation must:
- Score the same window for both forward and reverse at each position.
- NOT reverse the score track after scanning (the reverse track is indexed by window start, not by genomic coordinate on the reverse strand).
- The `best` strand mode takes per-position max of forward and reverse scores at the same index.

### 6.7 Site coordinates

Sites are reported with:
- `seq_index`: zero-based sequence index in the batch (Python convention, CLI output).
- `start`: zero-based position of the first nucleotide of the motif occurrence.
- `end`: zero-based position one past the last nucleotide (`start + motif_length`).
- `strand`: `"+"` for forward, `"-"` for reverse.
- For reverse-strand hits: `start` and `end` refer to the forward-strand coordinates of the window, NOT to the reverse-complement coordinates. The site string is the reverse complement of `seq[start:end]`.

### 6.8 Sorting and tie-breaking

Hit arrays are sorted by `(seq_index ascending, score descending, start ascending, strand_idx ascending)`. This is Python's `np.lexsort((strand_idx, start, -score, seq_index))`.

Julia must produce the same sort order. The `strand_idx` convention: `0 = forward (+)`, `1 = reverse (-)`.

## Alternatives considered

### A. One-based CLI output

Rejected: breaks backward compatibility with existing Python users' scripts and pipelines. The CLI output is a public contract.

### B. Different tie-breaking (e.g., larger overlap first)

The Python implementation uses a simple "first wins" for offset ties and a multi-key tiebreaker for profile shifts. A more "meaningful" policy (e.g., prefer larger overlap, prefer center alignment) could be an improvement, but it would be a documented divergence requiring an ADR update and migration note. For now, Julia matches Python exactly.

### C. Reverse-complement track indexed by reverse-strand coordinate

Rejected: would change the `best` strand computation and complicate the scanning kernel. The Python convention (same window start for both strands) is simpler and already frozen by fixtures.

## Consequences

- Internal Julia code uses one-based indexing; the serializer converts to zero-based for CLI output.
- Offset and orientation values in CLI output are identical to Python.
- Reverse-complement scoring semantics are frozen: same window, complemented/reversed read order.
- Site coordinates are zero-based in CLI output, one-based internally (converted at serialization).
- Tie-breaking is deterministic and independent of thread count.

## Migration impact

- Python users: CLI output is unchanged. JSON schema preserves `start`, `end`, `orientation`, `offset` with the same semantics.
- Julia library users: `SiteHit` struct uses one-based `UnitRange{Int}` internally. Conversion to zero-based CLI output is in the serializer.
- `ComparisonResult.offset` is a plain `Int` with the same meaning in both Python and Julia (no indexing conversion needed — it's a relative displacement).