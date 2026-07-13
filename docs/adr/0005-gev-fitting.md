# ADR 0005: GEV Fitting

## Status

Accepted and implemented with a native Float64 fitter.

## Context

Null comparison scores are modeled with a Generalized Extreme Value (GEV)
distribution. The package must not require Python or SciPy at runtime, while
remaining numerically comparable with historical SciPy-derived fixtures.

## Decision

Implement GEV maximum-likelihood fitting in Julia using a deterministic BFGS
optimizer, numerical gradients, backtracking line search, method-of-moments
initialization, and explicit support checks.

`GEVFit` stores textbook shape `k`, location, scale, convergence state,
iteration count, and log likelihood. `GEVFitFailure` is returned for typed
degenerate or unsuccessful fits. `survival` and `cdf` use Float64 and stable
formulas near probability boundaries.

Julia's shape convention is the sign inverse of SciPy's `genextreme` shape:
`k = -c`. Native optimization is tolerance-compatible, not parameter
bit-identical. Survival probabilities are the primary scientific comparison;
parameter tolerances are maintained in focused tests.

There is no automatic empirical fallback and no PythonCall runtime extension.
Callers must handle `GEVFitFailure` explicitly.

## Consequences

- GEV fitting and survival calculations remain Float64 even though scan and
  alignment values are Float32.
- Non-finite samples, degenerate inputs, invalid support, and non-positive scale
  fail through typed results rather than silent substitution.
- Stored null metadata records textbook `(shape, location, scale)` values.
- Compatibility fixtures may not be regenerated solely to hide an optimizer
  regression.
