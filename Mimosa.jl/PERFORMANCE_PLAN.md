# Mimosa.jl Performance and Parallelism Implementation Plan

## 1. Objective

Reduce the end-to-end latency and allocation volume of profile comparison,
especially the representative workload of one prepared query versus 50 targets
on 10,000 sequences of length 100. Add deterministic top-level parallelism to
the public one-to-many API and remove repeated scanning and profile preparation
from comparison and null-distribution workflows.

This plan is implementation-ready. Complete phases in order because later
parallel speedups depend on removing the current allocation and memory-bandwidth
bottlenecks first.

The primary target pipeline is:

```text
raw model or ScoreProfile
    -> scan, when required
    -> empirical normalization
    -> PreparedProfile
    -> allocation-light profile alignment
    -> deterministic one-to-many reduction
```

## 2. Current Measured Baseline

Measurements were taken with Julia 1.12.6 on the workload shape used by
`benchmark/bench_1v50.jl`: 10,000 rows, 86 scores per row, CO metric,
`search_range=10`, and `window_radius=5`.

| Operation | Median time | Allocated bytes |
|---|---:|---:|
| `flatten_bundle` | 0.5 ms | 6.9 MB |
| `fit(EmpiricalLogTail)` on duplicated strands | 27 ms | 26.8 MB |
| `normalize_bundle` | 239 ms | 7.0 MB |
| `profile_compare` on prepared profiles | 94-105 ms | 159.9 MB |
| `compare(prepared_query, raw_target)` | 370 ms | 202.1 MB |
| current serial PWM batch scan | 15.1 ms | 7.6 MB |
| current threaded PWM batch scan, 4 threads | 7.4 ms | 7.6 MB |

Validated prototypes produced:

| Prototype | Median time | Allocated bytes | Result |
|---|---:|---:|---|
| direct empirical rank transform | 68 ms | 17.2 MB | exactly equal normalized data |
| scratch-buffer profile alignment, four orientations | 47 ms | 73 KB | exactly equal CO result |
| direct-flat serial PWM scan | 13.4 ms | 3.5 MB | identical scores |
| direct-flat threaded PWM scan, 4 threads | 5.2 ms | 3.5 MB | identical scores |
| outer target parallelism, 4 threads | 2.24x speedup | n/a | ordered results |

The current `@code_warntype` results for concrete profile entry points are type
stable. Type instability is not the primary bottleneck. The dominant issues are
algorithmic duplication, binary lookups, and temporary allocation.

## 3. Required Semantics and Working Rules

1. Preserve the Python-compatible normalization, anchor, shift, orientation,
   site-count, and tie-breaking semantics.
2. Preserve the accumulation order inside one `(orientation, shift)` score.
   Parallelize independent targets or independent shifts, not the row reduction.
3. Serial and threaded results must have the same order. Scores, offsets,
   orientations, metrics, and site counts must be identical unless an existing
   compatibility test already uses a documented tolerance.
4. Do not add LoopVectorization, Polyester, FLoops, or another dependency in the
   first implementation. Base Julia and the existing execution-policy layer are
   sufficient for the measured bottlenecks.
5. Do not use nested parallelism. When targets are processed in parallel, each
   target scan and comparison must use `SerialExecution()` internally.
6. Preserve the public `fit`, `normalize_bundle`, `profile_compare`, and serial
   `compare` behavior. Add optimized internal paths rather than changing public
   semantics.
7. Before editing, inspect `git status`. Do not overwrite or revert the existing
   user changes in `benchmark/README.md`, `benchmark/bench_1v50.jl`, or
   `src/statistics/null_distribution.jl`.
8. Add tests in the same phase as each source change. Do not defer correctness
   tests until the threading phase.
9. Run JuliaFormatter using the repository test environment after every phase.
10. Record before/after benchmark JSON on the same machine and thread count.

## 4. Target Public API

Add the following supported methods while retaining existing methods:

```julia
compare(
    query::PreparedProfile,
    target::PreparedProfile;
    metric=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
)::ComparisonResult

compare(
    query::PreparedProfile,
    targets::AbstractVector{<:ScoreProfile};
    execution::ExecutionPolicy=SerialExecution(),
    metric=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
    min_logfpr::Union{Nothing,Real}=nothing,
)::Vector{ComparisonResult}

compare(
    query::PreparedProfile,
    targets::AbstractVector{<:PreparedProfile};
    execution::ExecutionPolicy=SerialExecution(),
    metric=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
)::Vector{ComparisonResult}
```

Also add `execution::ExecutionPolicy=SerialExecution()` to model-based
`prepare_profile` and profile-resolution methods so callers can explicitly
parallelize a single scan. The one-to-many methods must override inner execution
to `SerialExecution()` when outer target parallelism is active.

`PreparedProfile` must store the `min_logfpr` value used to build its anchors.
Prepared-to-prepared comparison must reject profiles prepared with different
thresholds. For a prepared query and raw target, `min_logfpr=nothing` means use
the query's stored threshold; an explicitly different threshold must raise
`ArgumentError` rather than silently mixing incompatible anchor sets.

## 5. Phase 0: Correct and Split the Benchmarks

### Source files

- `benchmark/bench_1v50.jl`
- `benchmark/runbenchmarks.jl`
- `benchmark/README.md`
- `benchmark/baseline.json`

### Required changes

- [ ] Rename the current "target precompute (scan + profile)" stage to "target
      scan + ScoreProfile construction". It does not call `prepare_profile`.
- [ ] Rename the current "pure profile comparison" measurement. It includes
      target normalization and anchor collection through
      `compare(::PreparedProfile, ::ScoreProfile)`.
- [ ] Add separate benchmark stages for:
      `scan`, `fit`, `normalize`, `collect anchors`, `prepare_profile`,
      `profile_compare` with both sides prepared, serial one-to-many with raw
      targets, serial one-to-many with prepared targets, threaded one-to-many
      with raw targets, and threaded one-to-many with prepared targets.
- [ ] Pass `execution=ThreadedExecution()` explicitly in threaded measurements.
      Setting `JULIA_NUM_THREADS` alone must not be described as enabling an API
      that still defaults to `SerialExecution()`.
- [ ] Record `execution`, `nthreads`, target state (`raw` or `prepared`),
      `min_logfpr`, `search_range`, and `window_radius` in benchmark parameters.
- [ ] Add allocation and memory columns to `bench_1v50.jl` output or migrate the
      workload into BenchmarkTools in `runbenchmarks.jl`.
- [ ] Keep one end-to-end fused measurement that scans, prepares, and compares
      each target. Do not call this a pure comparison.
- [ ] Generate a non-empty controlled-machine `baseline.json` only after all
      implementation phases are complete. Do not store prototype numbers as the
      release baseline.

### Acceptance criteria

- Every reported stage corresponds to exactly one named API operation.
- A 1-thread and 4-thread run visibly use different `ExecutionPolicy` values.
- The JSON output can distinguish preparation improvements from alignment and
  threading improvements.

## 6. Phase 1: Add Prepared-to-Prepared Comparison and Batch APIs

### Source files

- `src/profiles/alignment.jl`
- `src/comparison/profile_comparison.jl`
- `src/comparison/results.jl`
- `src/Mimosa.jl` only if exports change
- `test/unit/test_profiles.jl`
- `test/unit/test_parallel.jl`
- `test/jet/test_jet.jl`
- `test/downstream/runtests.jl`

### Required changes

- [ ] Extract construction of `ProfileConfig` and final `ComparisonResult` from
      repeated `compare` methods into small typed internal helpers.
- [ ] Add `min_logfpr::Float32` to `PreparedProfile`. Retain a three-argument
      outer constructor that defaults to `0.0f0` if direct construction is part
      of the downstream contract, while all `prepare_profile` methods must store
      their actual threshold explicitly.
- [ ] Implement `compare(::PreparedProfile, ::PreparedProfile; ...)`. It must
      call `profile_compare` directly and must not normalize or collect anchors.
- [ ] Reject prepared-to-prepared comparison when the stored `min_logfpr`
      values differ. Do not recollect anchors implicitly in this method.
- [ ] Change scalar and vector prepared-query/raw-target methods so
      `min_logfpr=nothing` inherits `query.min_logfpr`; reject an explicitly
      different value because query anchors have already been fixed.
- [ ] Change the vector method from `Vector{ScoreProfile}` to
      `AbstractVector{<:ScoreProfile}`.
- [ ] Add the corresponding `AbstractVector{<:PreparedProfile}` method.
- [ ] Resolve the metric and construct `ProfileConfig` once outside the target
      loop. Do not parse a Symbol and construct the same config for every target.
- [ ] Preallocate `Vector{ComparisonResult}(undef, length(targets))` and write
      results by index for both serial and threaded execution.
- [ ] Use `_parallel_for` for the first implementation. Do not use a shared
      `push!`, shared scratch buffer, or unordered result collection.
- [ ] In the raw `ScoreProfile` vector method, prepare and compare a target in
      the same iteration. Do not retain all temporary prepared targets unless
      the caller explicitly passed prepared targets.
- [ ] Ensure exceptions from spawned target tasks propagate through `@sync` and
      do not leave a partially returned result vector.
- [ ] Preserve empty-vector behavior: return an empty `Vector{ComparisonResult}`
      without spawning tasks.

### Required tests

- Serial raw-target results equal the current scalar loop for every profile
  metric.
- Prepared-target results equal raw-target results.
- Prepared profiles retain their construction threshold; mismatched prepared
  thresholds and an incompatible explicit raw-target threshold are rejected.
- `SubArray` or another `AbstractVector` view of targets is accepted.
- Threaded results with `ntasks` 1, 2, and 4 equal serial results and retain
  target order.
- Empty and single-target vectors work under both policies.
- A failing target propagates the original exception.
- JET accepts scalar prepared-to-prepared and both vector entry points with a
  typed metric.

### Acceptance criteria

- Already prepared targets incur no normalization or anchor allocations.
- The 4-thread raw-target prototype speedup is reproducible at no less than
  1.7x on the controlled 4-core benchmark run.

## 7. Phase 2: Replace Redundant Empirical Normalization Work

### Source files

- `src/profiles/normalization.jl`
- `src/profiles/alignment.jl`
- `src/models/score_profile.jl`
- `test/unit/test_profiles.jl`
- `test/properties/test_properties.jl`
- `test/compatibility/test_profile_fixtures.jl`

### Required internal API

Implement an internal helper with a concrete input type:

```julia
_fit_transform_empirical(
    scores::RaggedArray{Float32},
)::Tuple{LogTailTable,RaggedArray{Float32}}
```

### Algorithm

1. Handle empty input exactly like the existing `fit` and `transform_scores`
   composition.
2. Compute `perm = sortperm(scores.data; rev=true)`.
3. Walk `perm` in descending score order and identify equal-score groups.
4. For a group occupying sorted indices `i:j`, compute the same cumulative tail
   value as the existing implementation:
   `Float32(-log10(Float64(j) / Float64(n)))`.
5. Assign that value to `normalized_data[perm[k]]` for every `k in i:j`.
6. Append one score and log-tail entry per group in descending order to the
   returned `LogTailTable`.
7. Return normalized data in original input order and a copied offsets vector.

### Required changes

- [ ] Use `_fit_transform_empirical` in `prepare_profile(::ScoreProfile)`.
- [ ] Keep public `fit` and `transform_scores` available and semantically
      unchanged for callers that fit on one dataset and transform another.
- [ ] Simplify `fit`: remove the `counts::Vector{Int}` allocation. Cumulative
      counts are the equal-score group end indices in the sorted array.
- [ ] Do not construct `flatten_bundle` for `ScoreProfile`, because its forward
      and reverse fields alias the same score data.
- [ ] Do not build a `Dict{Float32,Float32}` unless benchmarks prove it faster
      and its peak memory is acceptable. The validated `sortperm` approach is
      the default implementation.
- [ ] If non-finite scores are currently accepted by a public boundary, retain
      existing behavior or add an explicit validation change with separate
      tests and documentation. Do not silently change NaN ordering semantics as
      part of a performance patch.
- [ ] Retain the binary-search transform for a genuinely separate background
      dataset. A later optimization may use a sorted merge, but it is not needed
      for this phase.

### Required tests

- Compare the new table and normalized data with `fit` followed by
  `transform_scores` using exact equality for empty, singleton, all-equal,
  duplicate-heavy, `-0.0f0`, sorted, reverse-sorted, and random Float32 data.
- Add a randomized property test over ragged rows, including empty rows.
- Verify row offsets are equal and are not accidentally shared with mutable
  caller-owned offsets where the previous API copied them.
- Re-run frozen normalized-profile fixtures without regeneration.

### Acceptance criteria

- `_fit_transform_empirical` is at most 0.40x the time of the previous
  `flatten_bundle + fit + normalize_bundle` ScoreProfile path on the controlled
  10,000 x 86 workload.
- Peak allocated bytes are lower than the old path.
- Normalized data and `LogTailTable` contents are exactly equal for finite
  Float32 inputs.

## 8. Phase 3: Preserve and Exploit Strand Symmetry

### Source files

- `src/models/score_profile.jl`
- `src/profiles/normalization.jl`
- `src/profiles/anchors.jl`
- `src/profiles/alignment.jl`
- `test/unit/test_profiles.jl`
- `test/properties/test_properties.jl`

### Required changes

- [ ] In `prepare_profile(::ScoreProfile)`, create one normalized
      `RaggedArray` and construct `StrandPair(normalized, normalized)`. Preserve
      identity: `prepared.bundle.forward === prepared.bundle.reverse`.
- [ ] Update `_collect_both_anchors` to detect identical strand objects. Collect
      once and return `(anchors, anchors)` when the bundle is symmetric.
- [ ] Replace the unconditional four-orientation loop with fixed tuples selected
      from strand identity:

```text
both symmetric:       ++
query symmetric:      ++, +-
target symmetric:     ++, -+
neither symmetric:    ++, +-, -+, --
```

- [ ] Keep these orientation sets as compile-time tuples, not allocated vectors.
- [ ] Preserve global orientation priority `++ > +- > -+ > --` and all existing
      score/site/shift tie-breaking.
- [ ] Do not infer scientific strand symmetry from equal values. Only use object
      identity or an explicit future representation flag. An accidental value
      equality must not suppress orientations.

### Required tests

- Prepared `ScoreProfile` strands and anchor objects are identical by `===`.
- Symmetric optimized results equal a forced four-orientation reference path.
- Cover both symmetric, only query symmetric, only target symmetric, and neither
  symmetric.
- Add tie cases proving the returned orientation remains the earliest legal
  orientation under the existing priority.
- Test all five profile metrics and both best-anchor and threshold-anchor modes.

### Acceptance criteria

- A symmetric ScoreProfile-to-ScoreProfile comparison evaluates one orientation.
- Prepared ScoreProfile memory no longer contains duplicate normalized strand
  data or duplicate anchor arrays.
- Compatibility fixtures and downstream result orientation remain unchanged.

## 9. Phase 4: Remove Per-Row Candidate Allocations

### Source files

- `src/profiles/alignment.jl`
- optionally a new cohesive `src/profiles/scratch.jl`, included from
  `src/profiles/profiles.jl`
- `test/unit/test_profiles.jl`
- `test/properties/test_properties.jl`
- `test/jet/test_jet.jl`

### Required internal design

Add one private scratch object per independent comparison worker:

```julia
mutable struct CandidateScratch
    candidates::Vector{Int}
    seen_epoch::Vector{UInt32}
    epoch::UInt32
end
```

The exact integer type may change after measurement, but the design must reuse
storage across rows and shifts.

### Required changes

- [ ] Replace `_collect_row_candidates` with an in-place
      `_collect_row_candidates!` that calls `empty!` on `candidates` and uses an
      epoch array instead of allocating `falses(len1)`.
- [ ] Size `seen_epoch` once from the maximum query row length. Grow only when a
      longer row is encountered.
- [ ] Increment the epoch once per row. On epoch overflow, zero `seen_epoch` and
      restart at one.
- [ ] Preserve candidate insertion order: query anchors first, followed by
      realigned target anchors not already seen. This is required to preserve
      floating-point accumulation order.
- [ ] Allocate scratch once in `_score_orientation_pair` or at the comparison
      worker boundary and reuse it across all shifts.
- [ ] Keep the public `score_shift` method by allocating local scratch and
      delegating to an internal scratch-accepting method.
- [ ] Ensure every thread owns its scratch object. Never store a mutable global
      scratch buffer.
- [ ] After the general scratch path is correct, add an optional specialized
      best-anchor path for `min_logfpr <= 0`. Each non-empty row has at most one
      query and one target anchor, so duplicate handling can use scalar positions
      without a candidate vector. Keep this specialization only if it produces a
      measured improvement and does not duplicate metric logic excessively.
- [ ] Apply the same allocation-light candidate traversal to pooled CO, pooled
      Dice, rowwise CO, rowwise Dice, and cosine.

### Required tests

- Compare candidate positions from old/reference and new in-place collectors for
  empty rows, duplicate anchors, out-of-window anchors, negative and positive
  shifts, and target realignment at row boundaries.
- Compare every metric result exactly or at its existing documented tolerance.
- Verify deterministic repeated results and serial/threaded equivalence.
- Add an allocation test using a warmed representative workload. Avoid a brittle
  zero-allocation assertion; enforce a generous upper bound below 1 MB for the
  current 10,000 x 86 prepared comparison.
- Add JET coverage for the scratch-accepting hot path.

### Acceptance criteria

- Four-orientation prepared comparison allocates less than 1 MB instead of
  approximately 160 MB on the controlled workload.
- Four-orientation CO time is at most 0.60x the old implementation.
- No candidate ordering or numerical compatibility regression occurs.

## 10. Phase 5: Remove Duplicate Scans and Propagate Execution Policy

### Source files

- `src/comparison/profile_comparison.jl`
- `src/profiles/alignment.jl`
- all model scan adapters that call profile resolution
- `src/cli.jl`
- `test/unit/test_profiles.jl`
- `test/unit/test_parallel.jl`
- `test/integration/test_cli.jl`

### Required changes

- [ ] In `_resolve_profile_bundle`, when `background_sequences === nothing`, set
      `bg_raw = raw`; do not scan `sequences` a second time.
- [ ] Apply the same reuse in `prepare_profile(::AbstractMotifModel, sequences)`.
- [ ] Also reuse `raw` when the explicit background object is identical to the
      sequence object by `===`. Do not use an O(n) equality comparison here.
- [ ] Add `execution::ExecutionPolicy=SerialExecution()` to model-based
      `prepare_profile` and `_resolve_profile_bundle`, and pass it to every
      `scan` call.
- [ ] Pass CLI `--threads` policy through preparation and comparison. Ensure the
      CLI does not merely start Julia with more threads while calling serial APIs.
- [ ] When a threaded outer one-to-many loop invokes preparation, explicitly
      pass `SerialExecution()` to inner scans.
- [ ] Add test instrumentation or a small counting test model proving that the
      no-background path scans once rather than twice.

### Acceptance criteria

- Model preparation with no separate background performs one BothStrands scan.
- Explicit separate background still performs the required second scan.
- Serial and threaded model-based preparation return identical prepared data.

## 11. Phase 6: Improve Scheduling and Enforce No Nested Parallelism

### Source files

- `src/parallel/parallel.jl`
- `src/scanning/pwm_scan.jl`
- `src/scanning/higher_order_scan.jl`
- `src/profiles/alignment.jl`
- `test/unit/test_parallel.jl`

### Required changes

- [ ] Resolve the mismatch between the comment claiming `_parallel_depth`
      protection and the implementation, which currently has no such guard.
- [ ] Implement a Julia 1.10-compatible task-local nesting guard, or remove the
      claim and enforce inner `SerialExecution()` explicitly at every parallel
      public boundary. The selected approach must be tested with a nested call.
- [ ] Keep the current static contiguous scheduler for equal-cost work if it is
      beneficial for cache locality.
- [ ] Add an internal dynamic scheduler for variable-cost targets. Implement it
      with at most `policy.ntasks` spawned tasks and an atomic next-index counter
      or equivalent bounded work queue. Do not use `Threads.@threads` in a way
      that ignores `ThreadedExecution.ntasks`.
- [ ] For ragged sequence scanning, prefer cost-weighted static chunks based on
      output positions, not sequence count. Compute per-sequence cost using the
      corresponding `npositions` function and partition prefix-summed work into
      approximately equal-cost chunks.
- [ ] Keep result writes indexed and independent. Scheduling order must not alter
      returned order.
- [ ] Do not parallelize the row accumulation inside `score_shift`.
- [ ] Optionally add shift-level parallelism for the single-pair latency case
      only after target-level threading is complete. Store results for shifts in
      fixed index order and run the existing deterministic selection afterward.

### Required tests

- `ntasks` is respected when it is smaller than `Threads.nthreads()`.
- Dynamic scheduling executes every index exactly once, including `n=0` and
  `n<ntasks`.
- A synthetic heavy-tail workload distributes long jobs across workers and is
  faster than contiguous equal-count chunks on a multi-thread test run.
- Nested threaded entry points do not create inner parallel workers.
- Exceptions from workers propagate and all synchronization scopes terminate.

### Acceptance criteria

- One-to-many uses bounded dynamic target scheduling.
- Ragged scan uses balanced work based on predicted scan positions.
- There is a tested, factual no-nested-parallelism policy.

## 12. Phase 7: Write Batch Scan Results Directly into Ragged Storage

### Source files

- `src/scanning/pwm_scan.jl`
- `src/scanning/higher_order_scan.jl`
- `src/sequences/ragged.jl` if a reusable allocation helper is justified
- `test/unit/test_parallel.jl`
- model-family scan tests

### Required changes

- [ ] Compute output offsets before scanning from each sequence length and model
      geometry.
- [ ] Allocate one flat output vector for ForwardOnly, ReverseOnly, or
      BestStrand, and two flat vectors for BothStrands.
- [ ] Pass disjoint views of the flat vectors to existing in-place scan kernels.
- [ ] Under threaded execution, each worker may write only within its indexed
      offset range.
- [ ] Remove `Vector{Vector{T}}` output rows and the final copying
      `build_ragged(out_rows)` step from PWM and higher-order batch scanning.
- [ ] Preserve zero-length rows as equal adjacent offsets and valid empty views.
- [ ] Review higher-order helper signatures. Replace `npos_fn::Function` and
      `scan_fn!::Function` with parameterized callable types or direct dispatch
      if JET shows dynamic dispatch in the row loop.
- [ ] Do not change the numerical inner kernels in this phase. SIMD, `muladd`,
      and weight-layout changes require separate compatibility evidence.

### Required tests

- Exact equality with the previous/reference result for every model family,
  strand policy, empty row, short row, mixed ragged batch, and fixed batch.
- Exact serial/threaded equality for 1, 2, and 4 tasks.
- Allocation bounds confirming removal of per-sequence output vectors.
- JET coverage for PWM and one higher-order direct-flat batch scan.

### Acceptance criteria

- PWM 10,000 x 100 serial allocation is no more than 55% of the old path.
- Threaded direct-flat scan is no slower than the old threaded scan on fixed and
  ragged representative workloads.
- No model-family compatibility fixture changes.

## 13. Phase 8: Reuse Prepared Profiles in Null Construction

### Source files

- `src/statistics/null_distribution.jl`
- `src/profiles/alignment.jl`
- `test/unit/test_null_distribution.jl`
- `test/unit/test_parallel.jl`
- `benchmark/runbenchmarks.jl`
- relevant null-distribution documentation

### Required changes

- [ ] Add an internal or public path that accepts already prepared models for
      null pair comparison. Pair comparison must use
      `compare(::PreparedProfile, ::PreparedProfile)`.
- [ ] Prepare each distinct model at most once per `(sequences, background,
      min_logfpr)` configuration instead of scanning and normalizing it for every
      eligible pair.
- [ ] Build prepared profiles in deterministic model order. They may be prepared
      in parallel by model, with serial inner scans.
- [ ] Compare eligible prepared pairs in the existing deterministic pair order.
- [ ] Account for memory explicitly. Add an API or internal threshold allowing
      callers to supply prepared profiles or disable full precomputation when the
      estimated normalized-profile memory is too large. Do not silently cache
      unbounded multi-gigabyte profile collections.
- [ ] If adding an automatic threshold, calculate the estimate from score counts
      and strand symmetry and document the default. Do not base it only on model
      count.
- [ ] Include profile-preparation configuration in any cache key. A profile
      prepared with a different background or `min_logfpr` is not reusable.

### Required tests

- Count scan/preparation calls and prove each distinct model is prepared once.
- Raw-model and prepared-model null results have identical pairs, raw scores,
  GEV inputs, skipped entries, and fingerprints.
- Serial/threaded equivalence and stable pair order.
- Memory-threshold fallback produces the same result without retaining every
  prepared profile.

### Acceptance criteria

- Null construction complexity changes from preparation per pair to preparation
  per distinct model when precomputation is enabled.
- The null benchmark reports preparation and pair-comparison stages separately.

## 14. Phase 9: Documentation, Baseline, and Release Gates

### Required changes

- [ ] Update `benchmark/README.md` with corrected stage names and actual serial
      and threaded results.
- [ ] Document that `JULIA_NUM_THREADS` controls available threads while
      `ThreadedExecution` selects threaded library execution.
- [ ] Add one-to-many raw and prepared examples to README/API docs.
- [ ] Document the no-nested-parallelism rule and how `ntasks` is interpreted.
- [ ] Add a changelog entry for the new overloads and execution keyword. State
      whether the change is API-compatible.
- [ ] Generate and commit the controlled-machine `baseline.json` with non-empty
      results after compatibility and full-suite verification.

### Final performance gates

On the same controlled machine and workload:

- `prepare_profile(::ScoreProfile)` time is at most 0.40x the old baseline.
- Four-orientation prepared alignment allocation is below 1 MB.
- Four-orientation prepared alignment time is at most 0.60x the old baseline.
- Symmetric ScoreProfile comparison evaluates one orientation and is faster than
  the four-orientation path.
- Four-thread one-to-many raw comparison is at least 1.7x faster than serial.
- Four-thread one-to-many prepared comparison is at least 1.7x faster than
  serial.
- Serial optimized 1-vs-50 is materially faster than the current 19.3-second
  baseline; target at least 3x while preserving compatibility.
- No representative scan or non-profile benchmark regresses by more than 10%
  without documented profile evidence and approval.

## 15. Recommended Commit Sequence

Keep phases independently reviewable and bisectable:

1. `bench: split profile preparation and comparison measurements`
2. `perf: add prepared-to-prepared and execution-aware batch compare`
3. `perf: rank-normalize ScoreProfile data in one pass`
4. `perf: preserve ScoreProfile strand symmetry`
5. `perf: reuse profile candidate scratch buffers`
6. `perf: eliminate duplicate background scans`
7. `perf: add bounded dynamic and weighted scheduling`
8. `perf: scan directly into ragged output storage`
9. `perf: reuse prepared profiles during null construction`
10. `docs: publish performance results and controlled baseline`

Do not combine all phases into one commit. After each commit, run the focused
unit tests and the corresponding benchmark before proceeding.

## 16. Required Verification Commands

Run from the repository root with Julia available on `PATH`:

```bash
# Focused correctness during implementation
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=1 julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

# Static/type analysis through the repository test entry point
julia --project=Mimosa.jl/test Mimosa.jl/test/jet/test_jet.jl

# Formatting
julia --project=Mimosa.jl/test -e 'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'

# Downstream and docs
julia --project=Mimosa.jl/test/downstream Mimosa.jl/test/downstream/runtests.jl
julia --project=Mimosa.jl/docs Mimosa.jl/docs/make.jl

# Performance, always compare matching thread counts
JULIA_NUM_THREADS=1 julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/bench_1v50.jl
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/bench_1v50.jl
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --output /tmp/mimosa-after.json
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --baseline Mimosa.jl/benchmark/baseline.json
```

If `julia` is not on `PATH` in the current development environment, use the
installed launcher `/home/anton/.juliaup/bin/julia` for local verification only;
keep portable `julia` commands in repository documentation.

## 17. Definition of Done

This plan is complete only when all of the following are true:

- The benchmark labels accurately describe included work.
- Raw and prepared one-to-many APIs support explicit serial/threaded execution.
- Prepared-to-prepared comparison performs no normalization or anchor rebuild.
- ScoreProfile normalization avoids duplicated strands and per-value binary
  lookup when normalizing its own empirical sample.
- Candidate collection no longer allocates per row and shift.
- No-background model preparation scans once.
- Thread scheduling is bounded, load-aware, deterministic, and non-nested.
- Batch scanning writes directly into flat ragged storage.
- Null construction can prepare each distinct model once.
- Frozen compatibility fixtures, unit tests, property tests, JET, downstream
  tests, docs, and 1/4-thread test runs pass.
- Controlled before/after benchmark artifacts demonstrate all final performance
  gates without an unexplained regression elsewhere.
