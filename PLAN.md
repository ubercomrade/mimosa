# Joblib And CLI Migration Plan

## Goal

Move one-query-to-many-target comparisons to joblib full-pipeline parallelism
with disk-mmap cache, remove the prepared-profile LRU, and replace the `profile`
CLI command with `compare` and `compare-many`.

## API

1. Remove public `workers` and `parallel_prepare` parameters from `compare_many`.
2. Add `total_threads=1` and `inner_threads=1` parameters to `compare_many`.
3. Restrict `inner_threads` to integers from 1 through 4.
4. Require positive integer budgets and require `total_threads` to be divisible
   by `inner_threads`.
5. Derive `joblib_workers = total_threads // inner_threads`.
6. Run the complete target pipeline serially when `joblib_workers == 1`.
7. Run target preparation and alignment in joblib workers when
   `joblib_workers > 1`.
8. Set Numba threads to `inner_threads` in every joblib worker.
9. Support outer parallelism only for built-in models and prepared profiles.
   Require custom models to be prepared first or compared serially.
10. Preserve target input order in results.

## Disk Cache

1. Remove the prepared-profile LRU from `Cache`.
2. Remove memory-budget configuration, `OrderedDict`, byte accounting, and
   eviction helpers.
3. Keep cache format v4: load prepared profiles as read-only `np.memmap`
   arrays from disk.
4. Keep the legacy pickle read fallback.
5. Keep file locking for disk writes.
6. Retain per-worker cache state only for verified disk entries, not prepared
   profiles.
7. Clear stale per-worker state when the cache instance changes.
8. Do not create a cache or promise mmap reuse when `cache=None`.

## CLI

1. Remove `mimosa profile`.
2. Add `mimosa compare QUERY TARGET`.
3. Replace `--model1-type` and `--model2-type` with `--query-type` and
   `--target-type`.
4. Add `mimosa compare-many QUERY TARGET [TARGET ...]`.
5. Make all targets use one `--target-type`; query type is set separately with
   `--query-type`.
6. Add `--numba-threads` to `compare`, constrained to 1 through 4.
7. Add `--total-threads` and `--numba-threads` to `compare-many`.
8. Validate that the total budget is divisible by the Numba budget and derive
   joblib workers from the quotient.
9. Configure `NUMBA_NUM_THREADS` before importing modules that import Numba.
10. Keep shared metric, FASTA, background, cache, threshold, and alignment
    arguments consistent between both commands.
11. Emit one JSON object for `compare` and an ordered JSON array for
    `compare-many`.
12. Support p-value annotation for each `compare-many` result after validating
    the query and shared target types against the null distribution.

## Benchmark

1. Replace free-form worker options with comma-separated `--total-threads`
   and `--numba-threads` arguments.
2. Generate only divisible total/inner budget pairs.
3. Report total threads, Numba threads, and derived joblib workers.
4. Measure full-pipeline cold and disk-cache modes.
5. Remove the memory-cache benchmark mode.
6. Avoid preloading every prepared target during full-pipeline measurements.
7. Retain a separate prepared-only mode only if row-parallel alignment analysis
   remains useful.

## Tests

1. Rename CLI tests from `profile` to `compare`.
2. Test `compare-many` JSON output, target order, shared target type, and
   invalid budgets.
3. Test `total_threads=4, inner_threads=2`, `4x1`, `1x4`, and serial `1x1`.
4. Check serial and joblib equivalence for all metrics and threshold modes.
5. Reject custom raw models in the joblib path.
6. Remove LRU-specific tests.
7. Verify disk hits return read-only mmap-backed arrays without creating an
   in-memory prepared-profile store.
8. Verify cold and disk cache results are equal with joblib workers.
9. Add benchmark smoke coverage for `1x1`, `2x1`, `2x2`, and `4x1`.

## Documentation And Verification

1. Update README, CLI reference, quickstart, Python API, and storage docs.
2. Document `target_workers = total_threads / numba_threads` and the 1--4
   Numba-thread limit.
3. Describe disk mmap cache and removal of the process-local LRU.
4. Replace `mimosa profile` in all examples.
5. Run `uv lock` if dependencies change.
6. Run `uv run ruff check .`, `uv run pytest -q`, and
   `uv run bash examples/run.sh`.
7. Benchmark `1x1`, `1x4`, `2x2`, and `4x1` on 32, 48, and 64 targets.

## Open Decision

Decide whether `--cache-dir` remains optional or defaults to `.mimosa-cache`
for `compare` and `compare-many`.
