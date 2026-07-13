# ADR 0004: Parallelism and RNG

## Status

Accepted and implemented. Updated 2026-07-13 for bounded scheduling and nested
parallelism protection.

## Decision

Parallel execution is opt-in twice:

1. Julia starts with multiple runtime threads.
2. The API receives `ThreadedExecution(n)`.

`SerialExecution()` remains the default. `ThreadedExecution(n)` is capped by
the requested tasks, item count, and `Threads.nthreads()`.

Parallelize only the highest independent level:

| Workflow | Parallel unit |
|---|---|
| Batch scan | weighted sequence blocks |
| One-to-many comparison | targets |
| Null construction | eligible model pairs |
| Site selection/PFM reconstruction | sequences |

`_parallel_for` uses a bounded dynamic atomic queue for irregular items.
`_parallel_for_weighted` creates small contiguous approximate equal-cost blocks
for ragged sequence batches. A task-local nesting guard forces inner parallel
regions to execute serially.

Callers preallocate outputs and each worker writes only to its own indices.
Shared `push!`, shared scratch buffers, unordered collection, and floating-point
reductions across workers are forbidden. Worker exceptions propagate and no
partially populated public result is returned.

Do not parallelize row reductions inside profile alignment. Their operation
order is part of the numerical compatibility contract.

## RNG

Library workflows do not use global RNG state. Random sequence generation uses
an explicit seed. The Julia generator is intentionally not bit-compatible with
NumPy; cross-language historical comparisons use explicit FASTA or frozen
encoded data.

## CLI Consequences

`--threads=N` selects `ThreadedExecution(N)` but cannot create runtime threads.
The CLI rejects `N > Threads.nthreads()` and explains how to start Julia with
`--threads=N` or `JULIA_NUM_THREADS=N`. `build-null --jobs` is a deprecated
alias for `--threads` only.

Serial and threaded outputs must preserve ordering, exact discrete fields, and
the documented floating-point compatibility.
