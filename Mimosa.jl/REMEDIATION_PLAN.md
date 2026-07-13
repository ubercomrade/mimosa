# Mimosa.jl Profile-Only Remediation Plan

## 1. Objective

Convert Mimosa.jl into a profile-only motif comparison package and close the
architecture, correctness, CLI, reproducibility, and quality-gate issues found
during the release-candidate audit.

This is an intentional breaking change. Backward compatibility with the Julia
direct motif comparison API, its CLI, or motif-null bundles is explicitly out
of scope. Do not add deprecated aliases, compatibility shims, fallback dispatch,
or legacy schema readers for removed functionality.

The supported comparison pipeline after this work is:

```text
motif model or ScoreProfile
    -> strand-aware score profiles
    -> empirical normalization
    -> anchor collection and profile alignment
    -> profile metric
    -> ComparisonResult
```

## 2. Source Of Truth And Working Rules

1. This plan is the source of truth for this remediation slice.
2. Preserve the Python implementation as the scientific oracle; do not remove
   Python direct comparison code as part of the Julia cleanup.
3. Do not regenerate frozen fixtures unless Python reference behavior changes.
4. Remove Julia tests for intentionally removed behavior instead of changing
   them to expect errors or preserving dead APIs for their benefit.
5. Keep profile compatibility fixtures and compare intermediate normalized
   profiles, anchors, shifts, orientations, and final scores.
6. Implement vertical slices with code, tests, documentation, and feature-matrix
   updates in the same change.
7. Do not mark this remediation complete from unit tests alone. Run all commands
   in Section 11 and record the results.

## 3. Final Product Contract

### Supported

- PWM, PFM, BaMM, SiteGA, Dimont, Slim, and ScoreProfile input models.
- Scanning on forward, reverse, best, and both strands.
- Profile comparison for model/model, ScoreProfile/ScoreProfile, prepared/query,
  and one-to-many workflows.
- Profile metrics: `co`, `co_rowwise`, `dice`, `dice_rowwise`, and profile
  `cosine` (`CosineSimilarityProfile`).
- Profile-only null construction, GEV fitting, annotation, storage, cache, and
  deterministic serial/threaded execution.
- PFM reconstruction as an independent scanning/site-extraction API. It must not
  be presented as a direct comparison mode.

### Removed Without Compatibility

- `compare(query, target; metric=:pcc)` and every two-argument motif/PFM matrix
  comparison method.
- Column metrics and their names: `AbstractColumnMetric`, `PearsonCorrelation`,
  `EuclideanDistance`, direct `CosineSimilarity`, `parse_metric`,
  `score_columns`, `_column_pcc`, `_column_euclidean`, `_column_cosine`.
- Matrix-alignment types and functions: `Orientation`, `ORIENTATIONS`,
  `MotifCandidate`, `align_motif_matrices`, `score_motif_candidates`,
  `select_best`, and `prepare_motif`.
- The CLI `motif` command and all its options, help, parsing, runners, tests,
  examples, precompile workload, and benchmarks.
- `MotifNullStrategy`, motif-null construction, `strategy="motif"`, and direct
  comparison null benchmarks/tests.
- Old null bundles whose schema permits motif strategy. No legacy reader is
  required.
- Direct-comparison compatibility tests in the Julia test suite.

Profile `CosineSimilarityProfile` and the profile metric name `cosine` are not
part of the removed column-metric subsystem and must remain.

## 4. Phase A: Remove Direct Motif Comparison Completely

- [ ] Delete `src/comparison/metrics.jl`.
- [ ] Delete `src/comparison/alignment.jl`.
- [ ] Remove the direct comparison methods from `src/comparison/results.jl`;
      retain `ComparisonResult` in a small result-focused file.
- [ ] Update `src/comparison/comparison.jl` to include and export only profile
      metrics, result types, and profile comparison APIs.
- [ ] Remove every direct-comparison export from `src/Mimosa.jl` and component
      export lists. Use one authoritative export list in `src/Mimosa.jl` rather
      than duplicating public API declarations across include files.
- [ ] Remove `test/unit/test_metrics.jl` and `test/unit/test_alignment.jl` from
      the test tree and `test/runtests.jl`.
- [ ] Remove direct comparison sections from property, JET, type-stability,
      downstream, compatibility, serialization, precompile, and benchmark tests.
- [ ] Keep serialization tests by constructing `ComparisonResult` directly or
      obtaining it from profile comparison.
- [ ] Remove direct fixture consumption from
      `test/compatibility/test_oracle_fixtures.jl`. Do not delete shared root
      oracle files still used by the Python project.
- [ ] Remove all direct comparison examples and claims from README, API docs,
      quick start, architecture, downstream contract, release guide, feature
      matrix, changelog unreleased section, and C4/ADR descriptions where they
      describe the current Julia architecture.

Required residue check:

```bash
rg -n 'AbstractColumnMetric|PearsonCorrelation|EuclideanDistance|CosineSimilarity\b|parse_metric|score_columns|MotifCandidate|ORIENTATIONS|align_motif_matrices|score_motif_candidates|prepare_motif|metric=:pcc|metric=:ed' Mimosa.jl
```

The only acceptable `CosineSimilarity` match is the explicitly named profile
type `CosineSimilarityProfile`; historical changelog text may remain only when
clearly marked as removed in the breaking release.

## 5. Phase B: Make Null Construction Profile-Only

- [ ] Remove `MotifNullStrategy` and the motif-specialized `build_null` method.
- [ ] Decide whether a one-variant `NullStrategy` hierarchy adds value. Prefer
      removing it and storing an `AbstractProfileMetric` directly in a concrete
      `NullBuildConfig{M}` unless dispatch has a demonstrated second use.
- [ ] Make `build_null` require `sequences::EncodedSequenceBatch`; keep optional
      background sequences and profile alignment configuration.
- [ ] Change the default null metric to `OverlapCoefficient()` and reject all
      removed metric names at the CLI boundary.
- [ ] Remove the `strategy` keyword from the public `build_null` API. A fixed
      `"profile"` metadata value is redundant and should be removed unless it is
      needed for external provenance.
- [ ] Bump `NULL_FORMAT_VERSION` and the null schema version. Define a clean
      profile-only manifest; old motif-capable bundles must fail with a clear
      typed version error rather than being migrated automatically.
- [ ] Simplify null compatibility fingerprints around the actual inputs:
      profile metric/config, model collection, relations, sequences, and
      background. Do not retain a dead strategy fingerprint.
- [ ] Update `savenull`, `loadnull`, annotation compatibility checks, CLI output,
      conversion scripts, security corpus, docs, and fixtures for the new schema.
- [ ] Replace motif-null tests with profile-null tests covering all model
      families, serial/threaded equivalence, insufficient relations, malformed
      bundles, incompatible sequence/background fingerprints, and failed GEV.

## 6. Phase C: Simplify And Type The CLI

- [ ] Remove `MOTIF_OPTIONS`, `MOTIF_FLAGS`, `MOTIF_METRICS`, `_run_motif`,
      `_print_motif_help`, `_ensure_pwm`, motif dispatch, and motif subprocess
      checks.
- [ ] Remove `--strategy` from `build-null`; the command is profile-only.
- [ ] Rename shared model-type constants so they do not imply a deleted motif
      command.
- [ ] Split the current `src/cli.jl` into focused files, for example:
      `cli/parser.jl`, `cli/config.jl`, `cli/profile.jl`, `cli/build_null.jl`,
      `cli/model_tools.jl`, and `cli/main.jl`.
- [ ] Introduce immutable typed configs for each command. Convert strings exactly
      once at the parser/config boundary; runners must not read numeric values
      from `Dict{String,String}`.
- [ ] Centralize helpers for required strings, bounded integers, finite floats,
      paths, flags, and execution policy. Every invalid user value must produce
      `CLIError` and exit code 1, not `MethodError` and exit code 2.
- [ ] Validate positive sequence counts and lengths, non-negative search/window
      values, finite thresholds/background values, valid model indices, and
      mutually exclusive options.
- [ ] Wire `--threads` through profile preparation, scanning, comparison, and
      profile-null construction. Do not advertise threading where the public API
      cannot accept an `ExecutionPolicy`.
- [ ] Remove the unused `--jobs` alias unless there is current downstream demand;
      no backward compatibility is required.
- [ ] Keep stdout strictly machine-readable and diagnostics on stderr.

## 7. Phase D: Correct Profile Semantics And API Invariants

- [ ] Fix profile orientation priority to `++ > +- > -+ > --`. Encode rank
      explicitly instead of relying on tuple iteration order.
- [ ] Add a regression test where `++` is not maximal and `--`, `+-`, and `-+`
      partially tie, proving that `+-` and then `-+` beat `--`.
- [ ] Add validating inner constructors for `ProfileConfig` and any new CLI/null
      configs. Reject negative search ranges, window radii, realignment windows,
      and non-finite thresholds at the library boundary.
- [ ] Preserve exact Python shift tie-breaking: score, site count, absolute
      shift, then earlier negative-to-positive traversal.
- [ ] Replace incomplete pointer equality in `reverse_complement!` with a safe
      alias policy based on `Base.mightalias`, or implement a correct overlapping
      in-place algorithm. Test identical arrays and partially overlapping views.
- [ ] Add `make_random_sequences(rng::AbstractRNG, n, len)` as the primary API.
      A seed convenience wrapper may remain, but scientific callers must be able
      to supply their RNG explicitly.
- [ ] Add RNG reproducibility and state-advancement tests without relying on the
      global RNG.

## 8. Phase E: Reduce Structural Redundancy

- [ ] Merge the four 16-line higher-order compatibility adapters
      (`bamm_scan.jl`, `sitega_scan.jl`, `dimont_scan.jl`, `slim_scan.jl`) into a
      single geometry/compatibility file, or remove family-specific
      `npositions_*` wrappers in favor of `npositions(model, seq_len)`. No
      compatibility requirement prevents removing them.
- [ ] Replace callback-based `_ho_scan_batch` plumbing with dispatch on strand
      and execution policies where this reduces `Function` arguments and dead
      `isa BothStrands` branches. Confirm with JET and benchmarks before keeping
      the refactor.
- [ ] Review the export surface. Keep end-user domain types and stable operations;
      stop exporting internal CSR builders, alignment kernels, raw sorting
      helpers, and low-level storage helpers unless the downstream contract
      demonstrates a real use.
- [ ] Keep large files when they represent one cohesive algorithm. Split by
      responsibility, not by line-count targets.

## 9. Phase F: Repair Quality And Performance Gates

- [ ] Run Aqua with default checks enabled. Any necessary exclusion must be
      narrow, documented next to the call, and backed by an issue; do not disable
      whole categories such as ambiguities, stale dependencies, or unbound args.
- [ ] Fix the formatter job to instantiate and use `Mimosa.jl/test`, where
      JuliaFormatter is a test-only dependency.
- [ ] Make the JET job run a dedicated JET entry point instead of rerunning the
      complete suite under a different job name.
- [ ] Make the security job run the security corpus directly rather than running
      the full suite twice and piping away its exit status/output.
- [ ] Retire `benchmark/benchmarks.jl` if `benchmark/runbenchmarks.jl` supersedes
      it. CI and documentation must invoke one benchmark runner.
- [ ] Generate a real controlled-machine baseline with commit, Julia, CPU,
      thread-count, timestamps, results, allocations, and memory data. An empty
      `results` array is not a regression baseline.
- [ ] Ensure scheduled/manual CI uploads the generated JSON and compares it with
      the stored baseline. Report confirmed regressions using the documented
      threshold policy.
- [ ] Re-run allocation assertions after removing direct comparison; replace
      obsolete direct-metric benchmark coverage with representative profile
      workloads for all model families and site-density regimes.

## 10. Documentation And Release Accounting

- [ ] Add a breaking-change section to `CHANGELOG.md` listing every removed API,
      CLI command, metric, null strategy, and bundle version.
- [ ] Update the README and quick start so the first comparison example uses
      explicit sequences and a profile metric.
- [ ] Update architecture documentation to describe a single profile comparison
      path rather than parallel motif/profile strategies.
- [ ] Update `docs/src/feature_matrix.md` factually. Delete removed features;
      do not mark them `deferred` or `not-porting` as if they remained part of
      the product contract.
- [ ] Update storage and security docs for the new null format and rejection of
      old bundles.
- [ ] Update the downstream contract and its separate test environment to expose
      only profile-based comparison.
- [ ] Update repository guidance (`AGENTS.md`) only after implementation and all
      gates are complete. It must not claim completion before verification.

## 11. Required Verification

Run from the repository root with the supported Julia executable on `PATH`:

```bash
# Main package, minimum and current Julia in CI
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.test()'
JULIA_NUM_THREADS=1 julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

# Formatter in the test-only environment
julia --project=Mimosa.jl/test -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'

# Documentation and downstream consumer
julia --project=Mimosa.jl/docs Mimosa.jl/docs/make.jl
julia --project=Mimosa.jl/test/downstream Mimosa.jl/test/downstream/runtests.jl

# Benchmark metadata and representative profile workloads
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --report
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --baseline Mimosa.jl/benchmark/baseline.json
```

Also run targeted subprocess checks for `profile`, profile-only `build-null`,
`inspect-model`, `convert-model`, invalid numeric options, old null-bundle
rejection, and serial/threaded equivalence.

## 12. Definition Of Done

This remediation is complete only when all conditions hold:

- [ ] No direct motif comparison implementation, public symbol, CLI command,
      column metric, motif-null path, test, benchmark, or current-feature claim
      remains in Mimosa.jl.
- [ ] Profile comparison works for every supported model family and ScoreProfile.
- [ ] Profile orientation and shift tie-breaking match the Python oracle and ADR.
- [ ] All public configs validate their invariants before hot kernels run.
- [ ] `--threads` is either fully wired for every command that advertises it or
      removed from that command.
- [ ] Random public APIs accept `AbstractRNG`.
- [ ] Aqua, JET, formatter, docs, downstream, security, serial tests, threaded
      tests, compatibility tests, and subprocess CLI tests pass fail-closed.
- [ ] The benchmark baseline contains real results and the CI benchmark uses the
      canonical runner.
- [ ] Null format/version documentation matches bytes written and accepted by
      the implementation.
- [ ] Feature matrix, README, changelog, architecture, and `AGENTS.md` state only
      verified current behavior.

