# Numerical Compatibility

Mimosa.jl preserves scientific semantics through exact discrete contracts and
documented floating-point tolerances. Frozen fixtures must not be regenerated
only to make a regression pass.

## Exact Contracts

Encoded bytes, offsets, lengths, shapes, indices, counts, result order, pair
order, site coordinates, comparison offsets/orientations, and schema fields are
exact. Current schema versions are model 1, null 2, cache 1, and annotated
result 1.

## Floating-Point Contracts

Scan, normalization, anchor, and profile-alignment values are Float32. Metric
accumulators use Float64 where specified by the implementation and convert the
final score to Float32. Preserve operation order; do not use `@fastmath`,
reassociation, or parallel inner reductions.

Default cross-platform checks use `atol=1e-5`, `rtol=1e-4` for scan,
normalization, and profile scores unless a focused fixture documents a stricter
bound. Threshold, tie, index, and branch behavior remains exact.

GEV fitting and survival calculations use Float64. The native BFGS fit is
tolerance-compatible with historical SciPy-derived fixtures, not bit-identical.
Julia shape `k` is the sign inverse of SciPy `genextreme` shape `c`.

## Profile Comparison

The only model comparison pipeline is:

```text
scan -> empirical normalization -> anchors -> strand/shift alignment -> score
```

Metrics are `co`, `co_rowwise`, `dice`, `dice_rowwise`, and `cosine`.
`PreparedProfile` instances must have compatible `min_logfpr` thresholds.

Shifts run from negative to positive. Ties are resolved by score, contributing
site count, smaller absolute shift, then orientation priority `++`, `+-`, `-+`,
`--`; a complete tie retains the first visited shift.

## Geometry and Coordinates

Raw scan inputs use codes `0x00:0x04`. Public boundaries reject invalid codes,
unsupported axes, invalid geometry, undersized outputs, and aliased two-strand
destinations before unchecked kernels.

Julia site coordinates are one-based inclusive. CLI coordinates are zero-based
half-open. Forward and reverse scores at one position refer to the same physical
window.

## Reproducibility

Serial and threaded execution preserve ordering and exact discrete fields.
Julia and NumPy RNG streams are intentionally different; cross-language
historical comparisons use explicit FASTA or frozen encoded data. Stable
fingerprints use SHA-256 rather than Julia's `hash()`.
