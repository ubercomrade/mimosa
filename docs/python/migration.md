# Migrating to the JAX Implementation

These notes describe the current source tree relative to the earlier
Numba/joblib-based implementation. They do not imply that a new package version
has already been published. Check `mimosa --version` and the release notes of
the distribution you install.

## Installation and execution

- Python 3.12 and 3.13 are supported.
- JAX is required, not an optional acceleration path. The current dependency is
  `jax==0.10.0`; Numba and joblib are no longer runtime dependencies.
- The default installation provides CPU execution. GPU execution requires a
  compatible JAX plugin, runtime and driver; consult the
  [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
- `compare_many` processes one query and its targets in order in a single
  process. Its legacy `total_threads` and `inner_threads` arguments are
  validated but no longer control execution. They are not CLI options.
- Public preparation, comparison and Monte Carlo operations manage required
  Float64 calculations locally; callers do not need to change global JAX x64
  configuration for those operations.

For a reproducible checkout environment, run `uv sync --locked --group dev`
from the repository root. Use `uv run python` or `uv run mimosa` with that
environment.

## Removed and replacement interfaces

| Earlier interface | Current workflow |
|---|---|
| `HybridEmpiricalLogTail` and histogram `bins` | `EmpiricalLogTail`, the only supported strategy |
| `NullDistribution`, `build_null`, `annotate_results` | `direct_mc` for per-pair raw p-values; `adjusted_pvalues` and `evalue` for annotations |
| Shared null-bundle readers/writers and CLI `null` command | No pooled-null storage; use CLI `--pvalue` or save direct Monte Carlo results |
| Custom model implemented only through `scan` | Implement JAX-traceable `score_sites` on `MotifModel`, or use `WindowModel` |
| SiteGA scanning and bundle I/O | Temporarily disabled; existing files are not deleted |

These are incompatible API changes, not just a backend replacement. See the
[API reference](api.md) and [custom-model guide](extending.md) for current
signatures and runnable scorer examples.

## Comparison and statistical results

`compare` and `compare_many` use weighted Jaccard similarity on circular
profiles. Profiles prepared explicitly for comparison must use
`prepare_profile(..., circular=True)`. The default `circular=False` retains the
linear layout for other uses.

Normalization is now the threshold-independent exact empirical log-tail
strategy. `min_logerr` selects contributing positions without changing their
stored normalized values. Results can differ from the earlier hybrid
normalization and from reductions performed by the old backend.

Direct Monte Carlo randomizes the single **query**, not the targets:

1. Prepare the observed query and fixed targets.
2. Independently relabel the query in each replicate, rescan it, and fit that
   replicate's calibration table.
3. Reuse those query replicates for every target.
4. Count randomized scores greater than or equal to the observed score and
   return `(count + 1) / (k + 1)`.

`direct_mc(query, targets, sequences, *, k, seed, ...)` returns three arrays
with shape `(1, number_of_targets)`: observed scores, exceedance counts and raw
p-values. It accepts motif models, not score profiles or prepared profiles.
Custom models need a stable fingerprint as well as `score_sites`.

Query randomization differs from the earlier target-randomization protocol;
old p-values and pooled nulls are not interchangeable with these results.
The CLI records `annotation_schema_version: 2` and
`monte_carlo.version: "mimosa-direct-mc-v2"` with `randomized: "query"`.
BH-adjusted p-values and default E-values depend on the supplied target family.
See [Method and statistics](method.md) for assumptions and limitations.

JAX device selection and reduction order can affect floating-point results.
Do not assume bitwise identity between CPU and GPU, or reuse old timing
comparisons without separating compilation from warmed execution.

## Existing caches and models

The prepared-profile cache format is v8 and the algorithm version is 9.
Older prepared-profile formats are treated as misses; profiles and anchors
are rebuilt from the current inputs. Clearing a cache is optional for
correctness, but can reclaim space from old entries.

Portable PWM, BaMM, Dimont and Slim model bundles remain supported. SiteGA
bundles are retained on disk but are rejected while SiteGA is disabled.
Checksums detect corruption, not tampering: use trusted model and cache
directories.
