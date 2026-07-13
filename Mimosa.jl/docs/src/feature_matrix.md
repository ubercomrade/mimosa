# Feature Matrix

This document is the authoritative inventory of all user-facing capabilities in
Mimosa.jl, cross-referenced with test coverage and known limitations. Each row
specifies the Julia implementation status:

| Status | Meaning |
|--------|---------|
| `done` | Fully implemented and tested against frozen Python fixtures. |
| `partial` | Core path works; some options or edge cases are not yet covered. |
| `documented-divergence` | Intentional deviation from Python, documented in an ADR. |
| `deferred` | Planned but not yet implemented; owner and stage identified. |
| `not-porting` | Deliberately excluded from the Julia port with justification. |

---

## Model families

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| PWM (Position Weight Matrix) | `done` | `PWM{T,M,B}`, `readmodel`, `read_meme` | `test/unit/test_models.jl`, `test/compatibility/test_oracle_fixtures.jl` | None | — |
| PFM (Position Frequency Matrix) | `done` | `PFM{T,M}`, `read_pfm` | `test/unit/test_models.jl`, `test/compatibility/test_oracle_fixtures.jl` | None | — |
| BaMM (Bayesian Markov Model) | `done` | `BaMM{T,M}`, `read_bamm` | `test/unit/test_bamm.jl`, `test/compatibility/test_bamm_fixtures.jl` | None | — |
| SiteGA (dinucleotide model) | `done` | `SiteGA{T,M}`, `read_sitega` | `test/unit/test_sitega.jl`, `test/compatibility/test_sitega_fixtures.jl` | None | — |
| Dimont (Jstacs Bayesian network) | `done` | `Dimont{T,M}`, `read_dimont` | `test/unit/test_dimont.jl`, `test/compatibility/test_dimont_fixtures.jl` | None | — |
| Slim (Jstacs GenDisMix) | `done` | `Slim{T,M}`, `read_slim` | `test/unit/test_slim.jl`, `test/compatibility/test_slim_fixtures.jl` | None | — |
| ScoreProfile (precomputed scores) | `done` | `ScoreProfile`, `read_scores` | `test/unit/test_profiles.jl`, `test/compatibility/test_profile_fixtures.jl` | None | — |

## File formats

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| MEME (PWM input) | `done` | `read_meme`, `readmodel` | `test/unit/test_readers.jl` | Multi-motif MEME via `--query-index` / `--target-index` | — |
| PFM (whitespace-separated) | `done` | `read_pfm`, `readmodel` | `test/unit/test_readers.jl` | None | — |
| BaMM `.ihbcp` | `done` | `read_bamm`, `readmodel` | `test/unit/test_readers.jl`, `test/unit/test_bamm.jl` | None | — |
| SiteGA `.mat` | `done` | `read_sitega`, `readmodel` | `test/unit/test_readers.jl`, `test/unit/test_sitega.jl` | None | — |
| Dimont XML | `done` | `read_dimont`, `readmodel` | `test/unit/test_readers.jl`, `test/unit/test_dimont.jl` | Custom minimal XML parser (see Security docs) | — |
| Slim XML | `done` | `read_slim`, `readmodel` | `test/unit/test_readers.jl`, `test/unit/test_slim.jl` | Custom minimal XML parser (see Security docs) | — |
| Score FASTA | `done` | `read_scores` | `test/unit/test_readers.jl` | None | — |
| Portable bundle (TOML + NPY) | `done` | `writemodel`, `readmodel` | `test/unit/test_model_storage.jl`, `test/unit/test_null_storage.jl` | v1 schema; backward-compatible | — |
| SiteGA `.mat` writer | `done` | `write_sitega` | `test/unit/test_readers.jl` | Write-only; no round-trip for other formats | — |

## Strand policies

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| ForwardOnly | `done` | `ForwardOnly()`, `scan_forward!` | `test/unit/test_sequences.jl`, `test/compatibility/test_scan_fixtures.jl` | None | — |
| ReverseOnly | `done` | `ReverseOnly()`, `scan_reverse!` | `test/unit/test_sequences.jl` | None | — |
| BestStrand | `done` | `BestStrand()`, `scan_best!` | `test/unit/test_sequences.jl`, `test/compatibility/test_scan_fixtures.jl` | None | — |
| BothStrands | `done` | `BothStrands()`, `scan_both!` | `test/unit/test_sequences.jl`, `test/compatibility/test_scan_fixtures.jl` | None | — |

## Direct motif comparison

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| One-to-one PCC | `done` | `compare(q, t; metric=:pcc)` | `test/unit/test_metrics.jl`, `test/compatibility/test_oracle_fixtures.jl` | None | — |
| One-to-one Euclidean distance | `done` | `compare(q, t; metric=:ed)` | `test/unit/test_metrics.jl`, `test/compatibility/test_oracle_fixtures.jl` | None | — |
| One-to-one cosine similarity | `done` | `compare(q, t; metric=:cosine)` | `test/unit/test_metrics.jl`, `test/compatibility/test_oracle_fixtures.jl` | None | — |
| Cross-type comparison (via PFM reconstruction) | `done` | `compare(pfm1, pfm2; metric=...)` | `test/integration/test_cli.jl` | PFM reconstruction via scan + site extraction | — |
| All four orientations (++, +-, -+, --) | `done` | `compare` (automatic) | `test/unit/test_alignment.jl` | None | — |
| Deterministic tie-breaking | `done` | `compare` (built-in) | `test/unit/test_alignment.jl` | Per ADR 0006 | — |

## Profile comparison

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| One-to-one overlap coefficient (CO) | `done` | `compare(q, t, seq; metric=:co)` | `test/unit/test_profiles.jl`, `test/compatibility/test_profile_fixtures.jl` | None | — |
| One-to-one CO rowwise | `done` | `compare(q, t, seq; metric=:co_rowwise)` | `test/unit/test_profiles.jl` | None | — |
| One-to-one Dice similarity | `done` | `compare(q, t, seq; metric=:dice)` | `test/unit/test_profiles.jl` | None | — |
| One-to-one Dice rowwise | `done` | `compare(q, t, seq; metric=:dice_rowwise)` | `test/unit/test_profiles.jl` | None | — |
| One-to-one cosine (profile) | `done` | `compare(q, t, seq; metric=:cosine)` | `test/unit/test_profiles.jl` | None | — |
| One-to-many comparison | `done` | `compare(prepared, [sp...]; metric=...)` | `test/downstream/runtests.jl` | None | — |
| PreparedProfile (precomputation) | `done` | `prepare_profile(sp)` | `test/unit/test_profiles.jl` | None | — |
| Profile from ScoreProfile input | `done` | `compare(sp1, sp2; metric=...)` | `test/unit/test_profiles.jl` | None | — |
| Profile from motif scan | `done` | `compare(m1, m2, seq; metric=...)` | `test/compatibility/test_profile_fixtures.jl` | Requires FASTA or random sequences | — |

## Normalization

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| EmpiricalLogTail normalization | `done` | `EmpiricalLogTail()`, `fit`, `transform_scores` | `test/unit/test_profiles.jl` | None | — |
| LogTailTable lookup | `done` | `LogTailTable`, `lookup_score` | `test/unit/test_profiles.jl` | None | — |
| min-logfpr threshold | `done` | `min_logfpr` kwarg in `compare` | `test/unit/test_profiles.jl` | None | — |
| Background normalization | `done` | `background` kwarg in `compare` | `test/unit/test_profiles.jl` | Optional; defaults to none | — |
| profile_bundle / flatten / normalize | `done` | `profile_bundle`, `flatten_bundle`, `normalize_bundle` | `test/unit/test_profiles.jl` | None | — |

## Site extraction

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| BestPerSequence selector | `done` | `BestPerSequence()`, `selectsites` | `test/unit/test_sites.jl`, `test/compatibility/test_sites_fixtures.jl` | None | — |
| ThresholdHits selector | `done` | `ThresholdHits(threshold)`, `selectsites` | `test/unit/test_sites.jl` | None | — |
| TopFractionHits selector | `done` | `TopFractionHits(fraction)`, `selectsites` | `test/unit/test_sites.jl` | None | — |
| Site collection | `done` | `SiteCollection`, `SiteHit` | `test/unit/test_sites.jl` | None | — |
| Sort hits | `done` | `sort_hits!` | `test/unit/test_sites.jl` | None | — |
| Select top fraction | `done` | `select_top_fraction` | `test/unit/test_sites.jl` | None | — |
| Site strings | `done` | `site_strings` | `test/unit/test_sites.jl` | None | — |
| extract_site_matrix | `done` | `extract_site_matrix` | `test/unit/test_sites.jl` | None | — |

## PFM reconstruction

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| Reconstruct PFM from sites | `done` | `reconstruct_pfm` | `test/unit/test_sites.jl`, `test/compatibility/test_sites_fixtures.jl` | None | — |
| Pseudocount support | `done` | `reconstruct_pfm(...; pseudocount=...)` | `test/unit/test_sites.jl` | None | — |
| Orientation correction | `done` | `reconstruct_pfm` (built-in) | `test/unit/test_sites.jl` | None | — |
| Build PCM | `done` | `build_pcm` | `test/unit/test_sites.jl` | None | — |
| PCM to PFM conversion | `done` | `pcm_to_pfm` | `test/unit/test_models.jl` | None | — |
| PFM to PWM conversion | `done` | `pfm_to_pwm`, `pwm_from_pfm` | `test/unit/test_models.jl` | None | — |

## Null distributions

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| Motif strategy null | `done` | `build_null(...; strategy="motif")`, `MotifNullStrategy` | `test/unit/test_null_distribution.jl` | None | — |
| Profile strategy null | `done` | `build_null(...; strategy="profile")`, `ProfileNullStrategy` | `test/unit/test_null_distribution.jl` | Requires `EncodedSequenceBatch` | — |
| Typed NullBuildConfig | `done` | `NullBuildConfig{S,M}` | `test/unit/test_null_distribution.jl` | Metric type validated at construction | — |
| GEV fitting | `done` | `fit_gev` | `test/unit/test_gev.jl`, `test/compatibility/test_gev_fixtures.jl` | None | — |
| GEV fit failure handling | `done` | `GEVFitFailure` | `test/unit/test_gev.jl` | None | — |
| BH FDR adjustment | `done` | `adjusted_pvalues` | `test/unit/test_pvalues.jl` | None | — |
| E-value computation | `done` | `evalue` | `test/unit/test_pvalues.jl` | None | — |
| P-value from GEV | `done` | `pvalue` | `test/unit/test_pvalues.jl` | None | — |
| Strict mode (fail on few targets) | `done` | `strict=true` in `build_null` | `test/unit/test_null_distribution.jl` | None | — |
| min_null_targets | `done` | `min_null_targets` in `build_null` | `test/unit/test_null_distribution.jl` | None | — |
| Serial = threaded equivalence | `done` | `execution=ThreadedExecution(n)` | `test/unit/test_parallel.jl`, `test/unit/test_null_distribution.jl` | None | — |
| Model/relation/sequence/background fingerprints | `done` | `model_collection_fingerprint`, `relation_fingerprint`, `sequence_fingerprint` | `test/unit/test_null_distribution.jl` | None | — |
| Empirical rank-based fallback for failed GEV | `deferred` | — | — | Not yet implemented; see PLAN_2.md §11 | Owner: TBD / Stage: post-RC |
| Explicit AbstractRNG API | `deferred` | — | — | Library uses `MersenneTwister` internally; thread-count-independent seed derivation planned | Owner: TBD / Stage: post-RC |
| Degenerate/NaN/Inf statistical corpus | `deferred` | — | — | Edge-case statistical tests not yet comprehensive | Owner: TBD / Stage: post-RC |
| Null compatibility lookup/search | `deferred` | — | — | No directory-based null search; explicit `--null-distribution` required | Owner: TBD / Stage: post-RC |

## Null storage

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| savenull (portable bundle) | `done` | `savenull` | `test/unit/test_null_storage.jl` | v1 schema; atomic staged write | — |
| loadnull (portable bundle) | `done` | `loadnull` | `test/unit/test_null_storage.jl` | v1 schema; checksum validated | — |
| Bundle format versioning | `done` | `NULL_FORMAT_VERSION = 1` | `test/unit/test_null_storage.jl` | Only v1 supported | — |
| Path traversal protection | `done` | (internal validation) | `test/unit/test_null_storage.jl` | Symlink escape, `..`, absolute paths blocked | — |
| Checksum validation | `done` | (internal validation) | `test/unit/test_null_storage.jl` | SHA-256 mandatory | — |
| Julia-native convert-null command | `deferred` | — | — | Not yet implemented; see PLAN_2.md §11 | Owner: TBD / Stage: post-RC |

## Result annotation

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| annotate_results | `done` | `annotate_results` | `test/unit/test_null_distribution.jl` | None | — |
| AnnotatedResult type | `done` | `AnnotatedResult` | `test/unit/test_null_distribution.jl` | None | — |
| Annotation schema version | `done` | `ANNOTATED_RESULT_SCHEMA_VERSION = 1` | `test/unit/test_null_distribution.jl` | Version 1; backward-compatible | — |
| CLI `--pvalue` annotation | `done` | CLI `--pvalue` flag | `test/integration/test_cli.jl` | Requires explicit `--null-distribution` | — |
| CLI `--effective-number-of-targets` | `done` | CLI `--effective-number-of-targets` | `test/integration/test_cli.jl` | None | — |
| Compatibility metadata check | `done` | `_validate_null_compatibility` (CLI) | `test/integration/test_cli.jl` | Strategy, metric, sequence and background fingerprints checked | — |

## Cache

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| Content-based cache | `done` | `Cache`, `cache_key`, `cache_get`, `cache_set` | `test/unit/test_cache.jl` | Explicit; no global singleton | — |
| Atomic writes | `done` | `cache_set` | `test/unit/test_cache.jl` | Temp file + rename | — |
| Checksum validation on load | `done` | `cache_get` | `test/unit/test_cache.jl` | SHA-256 validated | — |
| Corruption recovery | `done` | `cache_get` (cache miss on corruption) | `test/unit/test_cache.jl` | Corrupted files treated as cache misses | — |
| clearcache | `done` | `clearcache` | `test/unit/test_cache.jl` | None | — |
| Cache metadata | `done` | `cache_get_meta` | `test/unit/test_cache.jl` | None | — |
| Content fingerprint | `done` | `content_fingerprint`, `model_fingerprint`, `sequence_fingerprint` | `test/unit/test_cache.jl` | None | — |
| Cache integration in scan/compare/build-null | `deferred` | — | — | Cache exists but not wired into hot workflows; see PLAN_2.md §11 | Owner: TBD / Stage: post-RC |

## Parallelism

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| SerialExecution | `done` | `SerialExecution()` | `test/unit/test_parallel.jl` | Default | — |
| ThreadedExecution | `done` | `ThreadedExecution(n)` | `test/unit/test_parallel.jl` | Top-level only; inner kernels serial | — |
| Serial = threaded result equivalence | `done` | — | `test/unit/test_parallel.jl` | Pre-allocated slots; order independent of scheduling | — |
| Deterministic output across thread counts | `done` | — | `test/unit/test_parallel.jl` | Per ADR 0004 | — |

## Sequence encoding and manipulation

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| EncodedSequenceBatch | `done` | `EncodedSequenceBatch` | `test/unit/test_sequences.jl`, `test/unit/test_validation.jl` | Code validation `0:4` before `@inbounds` | — |
| FASTA reader | `done` | `read_fasta`, `readsequences` | `test/unit/test_sequences.jl` | None | — |
| Random sequence generation | `done` | `make_random_sequences` | `test/unit/test_sequences.jl` | Seeded; uses `MersenneTwister` | — |
| encode_base / encode_sequence | `done` | `encode_base`, `encode_sequence` | `test/unit/test_sequences.jl` | None | — |
| Reverse complement | `done` | `reverse_complement`, `reverse_complement!` | `test/unit/test_sequences.jl` | Aliasing checked in `!` version | — |
| to_padded / from_padded | `done` | `to_padded`, `from_padded` | `test/unit/test_sequences.jl` | Padding validation | — |
| RaggedArray | `done` | `RaggedArray`, `build_ragged` | `test/unit/test_sequences.jl` | None | — |
| Unsafe constructor (hot paths) | `done` | `_unsafe_encoded_batch` (internal) | `test/unit/test_validation.jl` | `Val{:unsafe}` token; internal only | — |

## CLI commands

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| `motif` | `done` | `main(["motif", ...])` | `test/integration/test_cli.jl` | None | — |
| `profile` | `done` | `main(["profile", ...])` | `test/integration/test_cli.jl` | None | — |
| `build-null` | `done` | `main(["build-null", ...])` | `test/integration/test_cli.jl` | None | — |
| `cache clear` | `done` | `main(["cache", "clear", ...])` | `test/integration/test_cli.jl` | None | — |
| `inspect-model` | `done` | `main(["inspect-model", ...])` | `test/integration/test_cli.jl` | None | — |
| `convert-model` | `done` | `main(["convert-model", ...])` | `test/integration/test_cli.jl` | None | — |
| `--help` / `--version` | `done` | `main(["--help"])`, `main(["--version"])` | `test/integration/test_cli.jl` | None | — |
| `--quiet` / `--verbose` | `done` | `--quiet`, `--verbose` flags | `test/integration/test_cli.jl` | Implemented; suppresses stderr info | — |
| `convert-null` CLI command | `deferred` | — | — | Not yet implemented; see PLAN_2.md §11 | Owner: TBD / Stage: post-RC |
| Interactive progress | `not-porting` | — | — | Deliberately excluded; CLI is batch-oriented | — |
| Debug stacktrace mode | `not-porting` | — | — | Deliberately excluded; `--verbose` provides diagnostics | — |

## CLI options

| Option | Commands | Status | Test ID | Notes |
|--------|----------|--------|---------|-------|
| `--model1-type` / `--model2-type` | motif, profile | `done` | `test/integration/test_cli.jl` | Required; validated against allowed types |
| `--metric` | motif, profile, build-null | `done` | `test/integration/test_cli.jl` | Validated per strategy |
| `--strategy` | build-null | `done` | `test/integration/test_cli.jl` | `motif` or `profile` |
| `--fasta` | motif, profile, build-null | `done` | `test/integration/test_cli.jl` | Profile strategy only for build-null |
| `--seed` | motif, profile, build-null | `done` | `test/integration/test_cli.jl` | Integer; tryparse validated |
| `--num-sequences` | motif, profile, build-null | `done` | `test/integration/test_cli.jl` | Positive integer |
| `--seq-length` | motif, profile, build-null | `done` | `test/integration/test_cli.jl` | Positive integer |
| `--search-range` | profile, build-null | `done` | `test/integration/test_cli.jl` | Non-negative integer |
| `--window-radius` | profile, build-null | `done` | `test/integration/test_cli.jl` | Non-negative integer |
| `--realign-window` | profile, build-null | `done` | `test/integration/test_cli.jl` | Non-negative integer |
| `--min-logfpr` | profile, build-null | `done` | `test/integration/test_cli.jl` | Finite float |
| `--background` / `--background-freq` | motif, profile | `done` | `test/integration/test_cli.jl` | Float32; validated |
| `--pfm-mode` | motif | `done` | `test/integration/test_cli.jl` | Forces PFM reconstruction |
| `--pfm-top-fraction` | motif | `done` | `test/integration/test_cli.jl` | Float64; validated |
| `--query-index` / `--target-index` | motif | `done` | `test/integration/test_cli.jl` | MEME motif index |
| `--pvalue` | motif, profile | `done` | `test/integration/test_cli.jl` | Requires `--null-distribution` |
| `--null-distribution` | motif, profile | `done` | `test/integration/test_cli.jl` | Portable null bundle path |
| `--effective-number-of-targets` | motif, profile | `done` | `test/integration/test_cli.jl` | Positive integer |
| `--strict` | build-null | `done` | `test/integration/test_cli.jl` | Fail on insufficient targets |
| `--min-null-targets` | build-null | `done` | `test/integration/test_cli.jl` | Positive integer |
| `--name-column` | build-null | `done` | `test/integration/test_cli.jl` | TSV column name; default: `motif` |
| `--group-column` | build-null | `done` | `test/integration/test_cli.jl` | TSV column name; default: `group` |
| `--ignore-missing` | build-null | `done` | `test/integration/test_cli.jl` | Flag; skip relations for unloaded motifs |
| `--model-type` | build-null | `done` | `test/integration/test_cli.jl` | Required; validated |
| `--groups` | build-null | `done` | `test/integration/test_cli.jl` | TSV/CSV path; required |
| `--output` | build-null | `done` | `test/integration/test_cli.jl` | Output directory; required |
| `--threads` / `--jobs` | motif, profile, build-null | `done` | `test/integration/test_cli.jl` | `--jobs` deprecated alias for `--threads` |
| `--cache-dir` | cache | `done` | `test/integration/test_cli.jl` | Default: `.mimosa-cache` |
| `--type` | inspect-model, convert-model | `done` | `test/integration/test_cli.jl` | Auto-detect if omitted |
| `--index` | inspect-model, convert-model | `done` | `test/integration/test_cli.jl` | MEME motif index |
| `--quiet` | all | `done` | `test/integration/test_cli.jl` | Suppresses informational stderr |
| `--verbose` | all | `done` | `test/integration/test_cli.jl` | Enables verbose stderr diagnostics |

## Legacy converters

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| convert-model (CLI) | `done` | `convert-model` command | `test/integration/test_cli.jl` | Legacy → portable bundle | — |
| readmodel auto-detect | `done` | `readmodel(path)` | `test/unit/test_readers.jl` | By file extension | — |
| writemodel | `done` | `writemodel(path, model)` | `test/unit/test_model_storage.jl` | Portable bundle format only | — |
| convert-null (CLI) | `deferred` | — | — | Not yet implemented | Owner: TBD / Stage: post-RC |

## Cross-language bundle exchange

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| Model bundle (TOML + NPY) | `done` | `writemodel`, `readmodel` | `test/unit/test_model_storage.jl` | Language-neutral; NPY is numpy-compatible | — |
| Null bundle (TOML + NPY) | `done` | `savenull`, `loadnull` | `test/unit/test_null_storage.jl` | Language-neutral; NPY is numpy-compatible | — |
| Cross-language exchange fixtures | `done` | — | `test/compatibility/test_oracle_fixtures.jl` | Python oracle produces same NPY blobs | — |

## Serialization

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| to_json | `done` | `to_json` | `test/unit/test_serialization.jl` | JSON to stdout | — |
| to_dict | `done` | `to_dict` | `test/unit/test_serialization.jl` | Dict for JSON serialization | — |
| AnnotatedResult serialization | `done` | `to_dict(annotated)` | `test/unit/test_serialization.jl` | Includes `annotation_schema_version` | — |

## Package and infrastructure

| Feature | Status | Entry point | Test ID | Limitations | Owner/Stage |
|---------|--------|-------------|---------|-------------|-------------|
| PrecompileTools workload | `done` | `src/precompile.jl` | — | Exercises representative paths during precompilation | — |
| Error hierarchy | `done` | `MimosaError`, `ModelFormatError`, `ModelDimensionError`, `InvariantError` | `test/unit/test_validation.jl` | Typed errors for model/storage/sequence | — |
| JuliaFormatter (BlueStyle) | `done` | `JuliaFormatter.format` | CI formatter job | Currently in runtime deps; should move to dev/test (PLAN_2.md C3) | — |
| Downstream contract test | `done` | `test/downstream/runtests.jl` | — | Needs separate Project.toml (PLAN_2.md C4) | — |
| Property-based testing | `done` | `test/properties/test_properties.jl` | — | Fuzzed encoded inputs, serial/threaded equivalence | — |

## Deferred and not-porting items (PLAN_2.md §11)

| Item | Status | Justification | Owner/Stage |
|------|--------|---------------|-------------|
| Empirical rank-based fallback for failed GEV fit | `deferred` | Planned; needs ADR for fallback policy | TBD / post-RC |
| Explicit `AbstractRNG` API and thread-count-independent seed | `deferred` | Library uses `MersenneTwister` internally; explicit RNG API planned | TBD / post-RC |
| Degenerate, constant, tiny, NaN/Inf, extreme-tail statistical corpus | `deferred` | Edge-case coverage planned | TBD / post-RC |
| Null compatibility lookup/search with explicit directories | `deferred` | No auto-discovery; explicit `--null-distribution` required | TBD / post-RC |
| Cache integration in scan/profile/compare/build-null hot workflows | `deferred` | Cache infrastructure exists; not wired into hot paths | TBD / post-RC |
| Cross-language model/null bundle exchange fixtures | `done` | Bundle format is language-neutral (TOML + NPY) | — |
| Julia-native `convert-null` command | `deferred` | Not yet implemented | TBD / post-RC |
| Interactive progress | `not-porting` | CLI is batch-oriented; no interactive prompts | — |
| Debug stacktrace mode | `not-porting` | `--verbose` provides diagnostics; no debug mode planned | — |
| DataFrames extension | `not-porting` | No real downstream demand; add only if needed | — |
| PackageCompiler app / static binary | `deferred` | Planned as separate artifact; not blocking library RC | TBD / post-RC |
| conda / Bioconda packaging | `deferred` | Document conda strategy without Python runtime dependency | TBD / post-RC |

## Test inventory

| Test file | Category | Lines (approx) |
|----------|----------|----------------|
| `test/unit/test_models.jl` | Model construction, PFM/PWM | ~200 |
| `test/unit/test_readers.jl` | File format parsing | ~150 |
| `test/unit/test_metrics.jl` | Column metrics (PCC, ED, cosine) | ~100 |
| `test/unit/test_alignment.jl` | Matrix alignment, tie-breaking | ~120 |
| `test/unit/test_serialization.jl` | JSON serialization | ~80 |
| `test/unit/test_sequences.jl` | Encoded sequences, FASTA, reverse complement | ~200 |
| `test/unit/test_profiles.jl` | Profile comparison, normalization | ~250 |
| `test/unit/test_sites.jl` | Site extraction, PFM reconstruction | ~200 |
| `test/unit/test_bamm.jl` | BaMM construction and scanning | ~150 |
| `test/unit/test_sitega.jl` | SiteGA construction and scanning | ~120 |
| `test/unit/test_dimont.jl` | Dimont construction and scanning | ~120 |
| `test/unit/test_slim.jl` | Slim construction and scanning | ~120 |
| `test/unit/test_gev.jl` | GEV fitting | ~100 |
| `test/unit/test_pvalues.jl` | P-value, BH FDR, E-value | ~80 |
| `test/unit/test_relations.jl` | Group relations parsing | ~100 |
| `test/unit/test_null_distribution.jl` | Null distribution build, strategy/metric validation | ~300 |
| `test/unit/test_null_storage.jl` | Null bundle save/load, hostile corpus | ~200 |
| `test/unit/test_parallel.jl` | Serial = threaded equivalence | ~100 |
| `test/unit/test_cache.jl` | Cache operations, corruption recovery | ~150 |
| `test/unit/test_model_storage.jl` | Model bundle save/load, round-trip | ~200 |
| `test/unit/test_validation.jl` | B2/B3: code validation, constructor invariants | ~237 tests |
| `test/properties/test_properties.jl` | Property-based tests, fuzzed inputs | ~100 |
| `test/compatibility/test_oracle_fixtures.jl` | Frozen Python fixtures: motif comparison | — |
| `test/compatibility/test_scan_fixtures.jl` | Frozen Python fixtures: scanning | — |
| `test/compatibility/test_profile_fixtures.jl` | Frozen Python fixtures: profile comparison | — |
| `test/compatibility/test_sites_fixtures.jl` | Frozen Python fixtures: site extraction | — |
| `test/compatibility/test_bamm_fixtures.jl` | Frozen Python fixtures: BaMM | — |
| `test/compatibility/test_sitega_fixtures.jl` | Frozen Python fixtures: SiteGA | — |
| `test/compatibility/test_dimont_fixtures.jl` | Frozen Python fixtures: Dimont | — |
| `test/compatibility/test_slim_fixtures.jl` | Frozen Python fixtures: Slim | — |
| `test/compatibility/test_gev_fixtures.jl` | Frozen Python fixtures: GEV fitting | — |
| `test/integration/test_cli.jl` | CLI integration: all commands, options | ~400 |
| `test/downstream/runtests.jl` | Downstream contract: public API only | ~150 |