# Numerical Compatibility

This document defines the scientific compatibility contract for the active
Julia implementation. Do not weaken tolerances or regenerate frozen fixtures
only to make a regression pass.

## Exact Values

The following must match exactly:

- encoded DNA bytes, row offsets, lengths, shapes, indices, and counts;
- model names and schema fields;
- result ordering and eligible null-pair ordering;
- comparison offsets and orientation labels;
- site coordinates and strand labels;
- model format version 1, null format version 2, cache format version 1, and
  annotated-result schema version 1.

## Floating-Point Values

Scanning, empirical normalization output, anchors, and profile alignment scores
are Float32. Profile metric accumulators use Float64 where the implementation
specifies it, then convert the final score to Float32. Preserve loop order and
do not introduce `@fastmath`, reassociation, or parallel reductions without
compatibility evidence.

Default cross-platform checks should use `atol=1e-5` and `rtol=1e-4` for scan,
normalization, and profile scores unless a focused test documents a stricter
bound. Boundary behavior around zero, thresholds, ties, and selected indices is
an exact semantic requirement even when neighboring floating-point values use a
tolerance.

GEV fitting and survival calculations use Float64. Native Julia fitting is
expected to be tolerance-compatible with SciPy-derived historical fixtures, not
bit-identical. Parameter and survival tolerances belong in the focused GEV tests
and must not be widened globally.

## Comparison Pipeline

Model comparison is always:

```text
scan both strands
  -> empirical tail normalization
  -> anchor collection
  -> shift/orientation profile alignment
  -> metric score
```

Supported metric names are `co`, `co_rowwise`, `dice`, `dice_rowwise`, and
`cosine`. A `PreparedProfile` owns its normalized strand bundle, anchor CSRs,
and `min_logfpr`; profiles prepared with incompatible thresholds must not be
compared or silently rebuilt.

Offset is query displacement relative to target. Positive means the query is
shifted right. Shifts are evaluated from `-search_range` to `+search_range`.
Ties are resolved by higher score, more contributing sites, smaller absolute
shift, then orientation priority `++`, `+-`, `-+`, `--`; a complete tie retains
the first visited shift.

## Coordinates and Reverse Complements

Julia library coordinates are one-based and inclusive. CLI serialization is
zero-based and half-open. Reverse-strand hits retain the forward-sequence window
coordinates while their extracted site string is reverse complemented.

At a given scan position, forward and reverse scores refer to the same physical
window. `BestStrand` takes the per-position maximum without reversing the score
track.

## Reproducibility

Serial and threaded execution must preserve result order and exact discrete
fields. Floating-point work is parallelized only across independent complete
items, never by changing reduction order within one item.

Julia and NumPy random generators are intentionally not bit-compatible. Use
explicit FASTA input or frozen encoded arrays for cross-language historical
comparisons. Cache and compatibility fingerprints use stable SHA-256 content
hashes, not Julia's session-dependent `hash()`.

Package-local parser/model inputs live under `Mimosa.jl/test/fixtures/`. The
former root Python oracle corpus is deleted in the current worktree while the
full test runner still references it; therefore the complete compatibility
suite is currently unavailable. Any deliberate replacement must document the
generator, dependency versions, command, checksums, and scientific reason.
