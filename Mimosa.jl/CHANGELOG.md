# Changelog

All notable changes to Mimosa.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking: Profile-Only Comparison

- Removed direct motif matrix comparison, its column metrics, alignment types,
  and the `motif` CLI command without compatibility aliases.
- Null construction now requires encoded sequences and profile metrics; motif
  null strategies and pre-v2 null bundles are rejected.
- Profile comparison uses `co`, `co_rowwise`, `dice`, `dice_rowwise`, and
  `cosine` through the strand-aware normalization pipeline.

### Added

- Execution-aware `selectsites` and `reconstruct_pfm` workflows.
- Bounded nested-parallelism protection and cost-weighted scan scheduling.
- Fused empirical profile normalization and target-level one-to-many comparison
  paths in Python and Julia, with serial/threaded equivalence coverage.
- **CITATION.cff** for citation support via GitHub and Zenodo.
- **CHANGELOG.md** tracking all notable changes through the migration and
  remediation process.

### Changed

- Higher-order batch scans now write directly into flat ragged storage instead
  of allocating and copying one vector per sequence.
- CLI and benchmarks now select `ThreadedExecution` explicitly and reject a
  requested CLI thread count that exceeds the Julia runtime thread count.
- Tests and benchmark documentation now follow the profile-only API, CLI, and
  null-storage contracts; the cross-language 1-vs-50 harness uses shared FASTA
  and MEME inputs.
- The cross-language profile benchmark now reports scan, normalization, anchor,
  alignment, prepared one-to-many, and end-to-end stages separately.

### Fixed

- Model-to-model profile comparison now converts the public `min_logfpr::Real`
  value to `Float32` before anchor construction, so the documented default call
  no longer raises a method error.

### Stage 10 — Release candidate preparation (PLAN_2.md remediation)

#### Added

- Julia CI matrix workflow for Linux x86_64 (Julia 1.10 minimum, latest stable)
  and macOS arm64 (latest stable), with serial and threaded configurations.
- Separate formatter, docs/doctests, Aqua, JET, subprocess CLI, and security
  corpus CI jobs.
- `test/downstream/Project.toml` — separate consumer environment for
  downstream contract verification without access to test internals.
- JET.jl as a test/dev dependency with targeted checks for public workflows
  and hot kernels.
- Full reproducible benchmark report with environment metadata (commit SHA,
  Julia version, dependency versions, CPU model, OS/kernel, RAM, thread count,
  input sizes, seeds, warm-up policy, median/min/variance, allocations, RSS,
  serial and multi-threaded scaling).
- Regression baseline with controlled-machine comparison and non-blocking
  CI comparison report.
- Feature matrix (`docs/feature_matrix.md`) with per-row status
  (`done`, `partial`, `documented-divergence`, `deferred`, `not-porting`),
  public Julia entry point, compatibility fixture/test ID, known limitations,
  and owner/stage for incomplete work.
- Doctests in quick-start documentation and CLI examples executed as
  subprocess integration tests.
- `CITATION.cff` for citation support.

#### Changed

- `JuliaFormatter` moved from runtime `[deps]` to test/dev tooling — no
  longer a runtime dependency.
- `docs/make.jl`: removed `warnonly=true` — documentation build fails on
  broken `@ref`/`@docs` references and doctest failures.
- `src/precompile.jl`: removed `try/catch` around precompile workload —
  precompilation failures now surface as errors instead of being silently
  suppressed.
- `test/runtests.jl`: Aqua quality checks are now fail-closed (no
  `try/catch` wrapper) — missing Aqua or Aqua test failure causes CI to fail.
- Package `Project.toml`: added `repo` field for General registry
  compatibility.
- README.md: removed outdated "Stage 1" status, fixed quick-start examples
  to match actual API signatures and CLI positional arguments.
- CLI documentation synchronized with actual help text and integration tests.
- Compat bounds for test and docs dependencies added.
- `Printf`, `Random`, `SHA`, `TOML` compat bounds adjusted to support
  Julia 1.10 (minimum supported version) without requiring 1.11+ stdlib
  versions.

#### Fixed

- `build-null` CLI now passes all parsed options (`--strategy`, `--metric`,
  `--fasta`, `--seed`, `--num-sequences`, `--seq-length`, profile alignment
  options, `--strict`, `--min-null-targets`, `--name-column`, `--group-column`,
  `--ignore-missing`) to the typed public API.
- `NullBuildConfig` is now typed: `NullBuildConfig{S<:NullStrategy,M}` with
  `MotifNullStrategy` and `ProfileNullStrategy` replacing string dispatch.
  Metric is validated at API boundary, not only in CLI.
- `build_null` dispatches on strategy type: motif strategy performs direct
  motif comparison; profile strategy explicitly accepts
  `EncodedSequenceBatch`, optional background, and profile configuration.
- `NullDistribution` metadata now contains actual model, relation, sequence,
  and background fingerprints — not placeholder values.
- CLI statistical annotation: `motif` and `profile` commands accept
  `--pvalue`, explicit `--null-distribution`, and
  `--effective-number-of-targets`; annotation checks compatibility metadata
  before applying; annotated JSON is marked with
  `annotation_schema_version = 1`.
- Bundle path traversal and symlink escape are now blocked; checksum is
  mandatory and strictly validated (`sha256:<64 lowercase hex>`); NPY
  header and payload length are parsed strictly before allocation;
  model-specific shape/order/span invariants are checked before reading blob.
- All public `EncodedSequenceBatch` constructors validate `0 <= code <= N_CODE`;
  internal unsafe constructor uses `Val{:unsafe}` token for hot paths only.
- `reverse_complement!` checks aliasing of dest/src.
- All scan kernels (`scan_forward!`, `scan_reverse!`, `scan_best!`,
  `scan_both!`, and all `_ho_scan_*!` kernels) validate `n_pos >= 0` and
  destination size before entering `@inbounds` regions.
- `extract_site_matrix` validates that the site window does not exceed
  sequence bounds.
- PFM constructor validates 4 rows, positive width, finite and non-negative
  values.
- PWM constructor validates background for finite, non-negative, and
  sum ≈ 1.0 (rtol = 1e-4).
- BaMM/SiteGA/Dimont/Slim constructors validate order/span >= 0 and <= 10
  (guard against exponentiation blow-up) before computing 5^(order+1).

#### Security

- Portable model and null bundles use bounded TOML/NPY boundary: v1 manifest
  and checksum are mandatory and typed; paths are validated before `realpath`
  and cannot escape bundle root; NPY headers and payload length are parsed
  strictly before allocation.
- Bundle writes are assembled in a sibling staging directory and committed
  via a single rename operation; orphan stages are never read.
- Hostile input tests cover traversal, symlink escape, checksum/version/type/
  size violations, malformed NPY, and staged-write cleanup.

#### Deprecated

- Nothing deprecated in this release cycle.

#### Removed

- `try/catch` suppression around Aqua quality checks — failures now propagate.
- `warnonly = true` from Documenter build — broken references and doctest
  failures now fail the docs build.
- `try/catch` around precompile workload — precompilation errors now surface.

### Stage 9 — Performance, latency, docs, and downstream contract

#### Added

- BenchmarkTools suite (`benchmark/benchmarks.jl`) covering PWM scan (widths
  8/15/30, lengths 100/200/1000, forward/best/rev_comp), batch scan
  (100/1000/10000 sequences, serial vs threaded), motif comparison (8×8 to
  15×30, all 3 metrics), higher-order scan (BaMM orders 0–3), site extraction
  and PFM reconstruction (100/1000 sequences), GEV fit (n=100/500/2000), BH
  FDR (n=1000), serial vs threaded equivalence, and CLI/serialization latency.
- PrecompileTools workload (`src/precompile.jl`) exercising representative
  code paths during package precompilation (zero I/O at `using Mimosa` time):
  PWM construction, scanning (all strand policies), batch scan, motif
  comparison (all metrics), site extraction, PFM reconstruction, GEV fit,
  BH FDR, JSON serialization, BaMM scan, cache fingerprints.
- Documenter documentation site — 14 pages: quick start, API reference, CLI
  guide, supported models, data layout, numerical compatibility,
  reproducibility, storage format, security, Python migration guide,
  extending Mimosa, downstream contract, architecture.
- Downstream contract test package (`test/downstream/runtests.jl`) with 26
  export checks and 18 workflow checks verifying full pipeline without
  access to internal submodules.
- ADR 0007 — CLI and distribution decision.
- `compare(::AbstractMotifModel, ::AbstractMotifModel, ::EncodedSequenceBatch)`
  for motif-derived profile comparison pipeline (scan → normalize → compare).
- `prepare_profile` / `PreparedProfile` for one-to-many profile comparison
  with query normalization/anchor preparation reused across targets.

#### Changed

- GEV fit optimized: allocations reduced from 59,504 → 120 (n=100), 118×
  faster (43 ms → 0.37 ms). Pre-allocated work vectors in
  `_numerical_gradient` and `_bfgs_optimize`.

### Stage 8 — CLI and legacy migration

#### Added

- Six CLI commands: `motif` (direct motif comparison for all model types),
  `profile` (profile-based comparison with score profiles and motif-derived
  profiles), `build-null` (null distribution build with threaded execution),
  `cache clear` (disk cache management), `inspect-model` (model metadata
  display), `convert-model` (legacy model to portable bundle conversion).
- Self-contained subcommand parser (0 dependencies, stdlib only).
- `app/mimosa.jl` — standalone CLI entry point.
- `scripts/convert_legacy_model.py` — trusted legacy model converter with
  `--trusted-input` security guard.
- `scripts/convert_legacy_null.py` — trusted legacy null converter with
  `--trusted-input` security guard.
- `make_random_sequences(n, len; seed)` — reproducible random DNA sequence
  generation for CLI fallback.
- Exit codes: 0 = success, 1 = usage error, 2 = runtime error.
- JSON output on stdout only; diagnostics and errors on stderr only.
- 56 CLI integration tests covering all commands, success/failure/help/
  missing args/malformed input/output files.

### Stage 7 — Parallelism, cache, and storage hardening

#### Added

- `ExecutionPolicy` abstract type with `SerialExecution` and
  `ThreadedExecution(ntasks)` typed execution policies (ADR 0004).
- `_parallel_for(f!, policy, n)` — generic parallel iteration helper with
  pre-allocated result slots and contiguous chunk partitioning.
- Parallel batch scanning for all model families (PWM, BaMM, SiteGA, Dimont,
  Slim) via shared `_ho_scan_batch` / `_ho_scan_batch_both` helpers.
- Parallel null build with pre-allocated raw_scores and pairs vectors.
- `Cache(directory; enabled=true)` — explicit, filesystem-backed cache with
  no global mutable singleton. Import never touches filesystem.
- Content-based cache keys: SHA-256 incorporating format version, algorithm
  name, algorithm version tag, and content parts.
- Atomic cache writes with checksum validation and corruption recovery
  (corrupted files → cache miss, not error).
- `clearcache(cache)` and `clearcache(cache, key)`.
- Portable model storage: `writemodel(path, model)` / `readmodel(path)` with
  bounded TOML manifest + NPY binary blobs, schema version 1, supporting all
  6 model families with strict checksum, path, NPY header/payload, and
  constructor-invariant validation.
- `readmodel` auto-detection: directory with `manifest.toml` → bundle;
  legacy file → format detection by extension.
- `readsequences(path; kwargs...)` — public alias for `read_fasta`.

### Stage 6 — Null distributions and statistics

#### Added

- `GEVFit` / `GEVFitFailure` — typed GEV fit result and typed failure for
  degenerate/constant/NaN/Inf samples.
- Native GEV MLE fit via custom BFGS optimizer (no LinearAlgebra dependency)
  with numerical gradient, backtracking line search, method-of-moments
  initialization, and support constraint validation (ADR 0005).
- `survival(gev, x)` — upper-tail SF with `-expm1` for precision.
- `benjamini_hochberg` (BH FDR) — `adjusted_pvalues(pvalues; method=BenjaminiHochberg())`.
- `evalue(pvalue, effective_n)` — E-value computation.
- `GroupRelations` — typed struct for motif group mapping and eligible pairs.
- `parse_group_relations` — TSV/CSV reader with delimiter sniffing.
- `NullDistribution` — typed struct with strategy, metric, fit, raw_scores,
  pairs, n_null, n_queries, skipped, and compatibility fingerprints.
- `NullPair` — typed contributing comparison pair.
- `build_null(models, relations; ...)` — null distribution build workflow.
- `AnnotatedResult` — comparison result enriched with significance fields.
- `annotate_results(results, dist; ...)` — annotate with null distribution.
- `savenull(path, dist)` / `loadnull(path)` — portable null storage (TOML +
  NPY, SHA-256 checksums, atomic writes).

### Stage 5 — Additional model families

#### Added

- **BaMM** (`BaMM{T,M}`): Bayesian Markov Model with higher-order context,
  `.ihbcp` parser, scanning via shared higher-order kernel, site extraction
  and PFM reconstruction.
- **SiteGA** (`SiteGA{T,M}`): Dinucleotide model with 25-row representation,
  `.mat` parser and writer, dinucleotide-specific scanning kernel.
- **Dimont** (`Dimont{T,M}`): Jstacs Bayesian network XML parser, dense 5-ary
  tensor materialization, scanning via shared higher-order kernel.
- **Slim** (`Slim{T,M}`): Jstacs GenDisMix classifier XML parser, log-sum-exp
  normalization, scanning via shared higher-order kernel.
- Shared higher-order scanning kernel (`_ho_scan_forward!` /
  `_ho_scan_reverse!`) parameterized by geometry (kmer, context, window,
  n_terms) for BaMM, Dimont, and Slim.

### Stage 4 — Sites and PFM reconstruction

#### Added

- `SiteHit` / `SiteCollection` typed structs and selectors.
- Best-per-sequence (`BestPerSequence`), threshold (`ThresholdHits`), and
  top-fraction (`TopFractionHits`) site selection.
- Reverse-strand sites extracted in canonical forward motif orientation.
- PFM reconstruction from selected sites with pseudocount and orientation
  correction.
- Stable ordering/tie-breaking for sites and minimum site behavior.

### Stage 3 — Profile comparison

#### Added

- `RaggedArray` and strand profile bundle (`StrandPair{RaggedArray{Float32}}`).
- `EmpiricalLogTail` normalizer with separate fit/apply API.
- Anchor collection: best anchors (`collect_best_anchors`) and threshold
  anchors (`collect_threshold_anchors`) with `AnchorCSR` for per-row access.
- Shift-based window alignment with target-anchor local realignment and
  full shift search (`score_shift`).
- Five profile metrics as typed metric types: `OverlapCoefficient` (`co`),
  `OverlapCoefficientRowwise` (`co_rowwise`), `DiceCoefficient` (`dice`),
  `DiceCoefficientRowwise` (`dice_rowwise`), `CosineSimilarity` (`cosine`).
- Four-orientation candidate search (`PROFILE_ORIENTATION_PAIRS`) with
  deterministic selection policy.
- One-to-many sequential path with `PreparedProfile` for query reuse.
- Motif-derived profiles (PWM scan → normalization → profile comparison).

### Stage 2 — Sequences and PWM scanning

#### Added

- `EncodedSequenceBatch` — validated typed sequence container with encoded
  A/C/G/T/N codes (0–4).
- `read_fasta` / `readsequences` — FASTA reader with lowercase, N/IUPAC
  handling.
- `reverse_complement!` — in-place reverse complement without temporary
  strings.
- `scan` / `scan!` — PWM scanning for single sequence and serial batch with
  zero per-position allocations in inner loop.
- `ForwardOnly`, `ReverseOnly`, `BestStrand`, `BothStrands` strand policies
  with typed scan results.

### Stage 1 — Package foundation and first PWM vertical slice

#### Added

- Julia package skeleton with `Project.toml`, module structure, and error
  hierarchy (`MimosaError`, `ModelFormatError`, `ModelDimensionError`,
  `InvariantError`).
- `PFM{T,M}` and `PWM{T,M,B}` concrete immutable parametric types.
- MEME and PFM file format parsers with limits and typed errors.
- PFM validation/conversion, PWM reverse complement, and score bounds.
- Direct PWM/PFM matrix alignment for all offsets and orientations.
- Three column metrics as typed metric types: `PearsonCorrelation` (`pcc`),
  `EuclideanDistance` (`ed`), `CosineSimilarity` (`cosine`).
- `ComparisonResult` typed result struct with stable JSON schema v1 draft.
- Tie-breaking policy from ADR 0006 (Python-frozen order: `++`, `+-`, `-+`,
  `--`).
- Metric direction, zero variance/norm, NaN, and minimum-overlap policy.
- JSON serializer (`to_json`, `to_dict`).

### Stage 0 — Audit and frozen oracle

#### Added

- Python reference architecture documentation.
- Feature matrix inventory of all user-facing capabilities.
- Numerical compatibility classes and tolerance policy.
- Format inventory with invariants for all supported file formats.
- 31 frozen oracle fixtures with versioned payloads and checksums.
- ADR 0001–0006: model type hierarchy, sequence representation, storage
  format, parallelism/RNG, GEV fitting, coordinate/offset/orientation
  conventions.
- Versioned oracle fixture generation script.

## [0.1.0] — 2026-07-13

Initial release candidate. Full migration of Python MIMOSA to an independent
Julia package with all six model families (PWM, PFM, BaMM, SiteGA, Dimont,
Slim), motif and profile comparison, site extraction, PFM reconstruction,
native GEV null distributions, portable TOML+NPY storage, content-based
cache, parallel execution, and thin CLI adapter.

### Known limitations

- Empirical rank-based fallback for failed GEV fit: deferred.
- Explicit `AbstractRNG` API for library-generated data: deferred
  (build_null does not generate data).
- Degenerate/NaN/Inf/extreme-tail statistical corpus expansion: deferred.
- Null compatibility lookup/search with explicit directories: deferred.
- Cache integration in scan/profile/compare/build-null hot workflows:
  deferred.
- Cross-language model/null bundle exchange fixtures: deferred.
- Julia-native `convert-null` command: not-porting (legacy nulls only in
  Python).
- Interactive progress and debug stacktrace mode: deferred.
- DataFrames extension: deferred (no downstream demand).
- PackageCompiler static binary: deferred.
- Conda/Bioconda packaging: deferred.
