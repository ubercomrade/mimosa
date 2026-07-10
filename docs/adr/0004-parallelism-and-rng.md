# ADR 0004: Parallelism and RNG

## Status

Proposed (Stage 0)

## Context

The Python implementation parallelizes via Numba's `@njit(parallel=True)` and `prange`, which distributes independent rows of the alignment kernel across threads. Thread count is controlled via `set_num_threads()` in a context manager (`_numba_thread_scope`). The `--jobs` CLI argument maps to Numba's thread count, not to Python-level parallelism.

This approach has several issues for Julia:
- Thread parallelism is embedded inside numerical kernels, making it non-composable.
- `set_num_threads()` mutates global state, which is not thread-safe.
- `prange` over rows means the kernel itself manages parallelism, preventing outer-level scheduling.
- Results may depend on thread count due to floating-point reduction non-associativity.
- No explicit RNG management: `np.random.default_rng(seed)` is used, but parallel null building has no per-task independent streams.

## Decision

### 4.1 Execution policy types

```julia
abstract type ExecutionPolicy end
struct SerialExecution <: ExecutionPolicy end
struct ThreadedExecution <: ExecutionPolicy
    ntasks::Int  # optional hint; defaults to Threads.nthreads()
end
```

### 4.2 Parallelism at the top level only

Numerical kernels (scanning, alignment, scoring) are **serial and composable**. Parallelism is applied at the outermost independent level:

| Operation | Parallel unit |
|---|---|
| Batch scan | Sequences (or sequence groups) |
| One-to-many comparison | Target models |
| Null distribution building | Query models (outer) × target groups (inner) |

No `@threads`, `@spawn`, or `Threads.@spawn` inside scan/alignment kernels. Kernels accept preallocated output buffers and write to indexed positions.

### 4.3 Deterministic output ordering

Results are stored in preallocated arrays indexed by the input order. Thread tasks write to their own indices; no `push!` to shared arrays. The output array is assembled after all tasks complete, in input order. Result order is independent of thread count and scheduling.

### 4.4 Floating-point determinism

Serial kernels produce deterministic results (same input → same output, same order). Parallel reduction is NOT used for floating-point scores. Each thread/task computes complete results for its assigned unit (sequence, model, pair). Results are not summed across threads — each unit's result is independent.

If a future optimization requires cross-thread reduction (e.g., pooled null score summation), it must:
1. Be proven necessary by profiling.
2. Use a deterministic reduction order (e.g., sort by task index before summing).
3. Be documented in an ADR update.
4. Pass serial/threaded equivalence tests.

### 4.5 RNG management

```julia
function build_null(rng::AbstractRNG, models, relations; kwargs...)
```

- All randomness accepts `AbstractRNG`.
- No global `rand()` or `Random.default_rng()` in library code.
- For parallel null building, derive independent RNG streams:
  ```julia
  subrng = FutureExpanding(rng, task_index)
  ```
  or hash-based derivation with a documented, version-stable hash function.
- RNG state does NOT depend on thread count. Each task gets a deterministic sub-RNG from `(base_seed, task_index)`.
- `hash()` is NOT used for seed derivation because Julia's `hash` is not guaranteed stable across versions. A dedicated stable hash (e.g., SHA-256 of a canonical string) is used.

### 4.6 BLAS control

If BLAS is used (unlikely for motif comparison, but possible for future extensions):
- `BLAS.set_num_threads(1)` when outer Julia threading is active.
- BLAS thread count is set once at module init or per-call, not mutated globally during computation.

## Alternatives considered

### A. Keep Numba-style inner parallelism

Rejected: non-composable, prevents outer scheduling, makes result order depend on thread count (due to reduction order), and is not idiomatic Julia.

### B. Distributed computing

Deferred: the problem sizes (thousands of motifs, millions of positions) fit in shared memory. Distributed computing adds complexity without benefit for the current scope. Could be added as an optional extension if needed.

### C. GPU

Deferred: requires benchmark evidence of need. Motif scanning is memory-bandwidth bound, not compute bound. GPU may help for very large batches, but this is a post-1.0 optimization.

## Consequences

- `--jobs` CLI flag is replaced by `--threads` (mapping to `Threads.nthreads()`).
- `SerialExecution` is the default; `ThreadedExecution` is opt-in.
- Users can run the same computation serially and threaded with identical results.
- `n_jobs` in `ComparatorConfig` is not carried to Julia. Thread control is an execution policy, not a comparison parameter.
- Random sequence generation in Julia will NOT produce the same bytes as Python's `np.random.default_rng`. Fixtures store encoded bytes; tests load them from disk.

## Migration impact

- Python `--jobs 4` → Julia `--threads 4` (or `JULIA_NUM_THREADS=4`).
- Python `config["n_jobs"]` → Julia `execution=ThreadedExecution(4)` parameter.
- Python `set_num_threads(n)` context → Julia `ThreadedExecution(n)` policy.
- Python `np.random.default_rng(seed)` → Julia `MersenneTwister(seed)` or `Xoshiro(seed)`. Seeds produce different sequences; fixtures store generated data.