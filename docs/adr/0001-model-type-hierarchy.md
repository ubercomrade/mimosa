# ADR 0001: Model Type Hierarchy

## Status

Accepted and implemented. Updated 2026-07-13 to reflect the current profile-only
comparison API and flat higher-order representations.

## Context

Mimosa supports six motif families with different invariants and scanning
geometry. Runtime string dispatch or a catch-all model containing `Any` would
hide those differences from the compiler and from API validation.

## Decision

Use a shallow hierarchy of concrete immutable parametric types:

```julia
abstract type AbstractMotifModel end
abstract type AbstractMatrixMotif <: AbstractMotifModel end
abstract type AbstractHigherOrderMotif <: AbstractMotifModel end
```

- `PFM` stores a 4-by-width frequency matrix.
- `PWM` stores a 5-by-width score matrix plus a four-base background tuple.
- `BaMM`, `Dimont`, and `Slim` store a flat
  `5^(order_or_span + 1)`-by-motif-length matrix and explicit order/span.
- `SiteGA` stores a 25-by-motif-length dinucleotide matrix.
- `ScoreProfile` is separate from `AbstractMotifModel`: it participates in
  comparison but cannot be scanned or used for site extraction.

Shared scanning geometry is exposed through dispatch on `motif_length`,
`window_size`, `scorematrix`, `scoretype`, and `site_start_offset`. Strings and
symbols are parsed only at API or I/O boundaries.

PFM and PWM remain distinct because their invariants and scientific meaning are
different. Higher-order matrices use Julia's two-dimensional column-major
layout in memory; portable NPY blobs are explicitly row-major on disk.

## Comparison Consequence

There is no direct model-matrix comparison. Two motif models are compared only
with an explicit `EncodedSequenceBatch` through scan, normalize, anchor, and
profile alignment. `ScoreProfile` values enter the same profile preparation and
alignment subsystem without scanning.

## Consequences

- Adding a model family requires a concrete type, constructor validation,
  scanning geometry, I/O, and focused tests.
- Heterogeneous collections require API-level function barriers; hot kernels
  dispatch on concrete types.
- Provenance belongs to I/O metadata and bundle manifests, not mutable model
  fields.
- Public constructors reject malformed dimensions, non-finite values, invalid
  backgrounds, and excessive order/span before dangerous allocation.
