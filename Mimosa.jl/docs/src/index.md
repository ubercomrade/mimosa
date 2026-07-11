# Mimosa.jl Documentation

## Data Layout

PWM weights use `(base, position)` layout (column-major Julia), where `base ∈ 1:5`
(A, C, G, T, N) and the fifth row holds the N-state score (per-column minimum).
PFM frequencies use `(base, position)` with `base ∈ 1:4`.

## Offset and Orientation Conventions

See [ADR 0006](../../docs/adr/0006-coordinate-offset-orientation-conventions.md)
for the full convention. Summary:

- Internal Julia indexing is one-based inclusive.
- CLI JSON output uses zero-based half-open coordinates.
- Offset is the displacement of the query relative to the target (positive =
  query shifted right). Iteration goes from negative to positive; first wins
  on equal score.
- Four orientation candidates: `++`, `+-`, `-+`, `--` with tie-break ranks
  0-3. Lower rank wins on equal score.
- Reverse complement of a PWM: flip base rows (A↔T, C↔G) and reverse position
  columns.

## Numerical Compatibility

- PFM-to-PWM conversion: `log((pfm + 1e-4) / background)`.
- PCC: zero-variance columns contribute 0 (denominator <= 1e-9 → 0).
- ED: `-mean(sqrt(sum((x-y)^2)))` per column; similarity (higher is better).
- Cosine: zero-norm columns contribute 0.

## Extension Boundary

New model families require a concrete struct and `compare`/`scorebounds` methods.
See `docs/src/extending_models.md` (planned).