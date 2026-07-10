# Python Reference Architecture

> **Stage 0 audit artifact.** Frozen at Python commit `95e8dbb` (2026-07-10).
> This document maps the existing Python implementation to guide the Julia port.
> It is not a specification of Julia behavior; it records what exists, what must be preserved, and what must not be copied literally.

## 1. Module map

The package `mimosa` lives under `src/mimosa/` and contains 10 070 lines across 35 Python files.

```
src/mimosa/
├── __init__.py          (75)   Public re-exports
├── api.py               (620)  High-level orchestration: configs, run_* entry points
├── batches.py           (232)  Padded sequence/masked score/profile batches (TypedDict)
├── cache.py             (133)  Profile cache: fingerprints, .npz atomic store/load
├── cli.py               (624)  Argparse CLI: profile, motif, build-null, cache
├── handlers.py           (282)  Built-in model handlers + registry registration
├── models.py             (142)  GenericModel dataclass, ModelHandler, registry
├── progress.py           (47)   tqdm wrapper, logging handler
├── scanning.py           (137)  Scan dispatch, strand resolution, score bounds
├── sites.py              (455)  Hit extraction, site tables, PFM reconstruction
├── types.py              (167)  ComparatorConfig, ComparisonResult, OneToOne/Many configs
├── validation.py          (95)  Input validation helpers
├── comparison/
│   ├── __init__.py       (25)   Re-exports
│   ├── common.py         (24)   Orientation tie-breaking, batch fingerprint
│   ├── config.py         (64)   Metric names, create_comparator_config
│   ├── motif.py          (279)  Direct matrix alignment strategy
│   ├── profile.py        (567)  Window-based profile comparison strategy
│   └── runner.py         (136)  Strategy registry, one-to-many, Numba thread scope
├── functions/
│   ├── __init__.py       (52)   Re-exports
│   ├── alignment.py      (431)  Fused Numba profile-window alignment kernels
│   ├── curves.py         (166)  ROC/PRC/pAUC (not used in comparison pipeline)
│   ├── formatting.py      (9)   format_params helper
│   ├── matrices.py        (17)   pfm_to_pwm, pcm_to_pfm
│   ├── profile.py        (150)  CO/Dice/cosine metrics, prepare_profile_bundle
│   ├── scanning.py       (319)  Numba scan kernels (forward/reverse/both)
│   └── tails.py          (171)  Empirical log-tail normalization
└── io/
    ├── __init__.py       (27)   Re-exports
    ├── bamm.py           (118)  BaMM .ihbcp reader
    ├── batches.py         (69)  FASTA + score-FASTA readers
    ├── dist.py            (16)  DIST writer (unused in main pipeline)
    ├── meme.py           (116)  MEME reader (single + multi-motif)
    ├── motifs.py          (21)  Compatibility facade
    ├── pfm.py             (38)  PFM reader/writer
    ├── sitega.py         (102)  SiteGA .mat reader/writer
    └── xml.py             (407)  Dimont/Slim XML readers
```

## 2. Subsystem responsibilities

### 2.1 Public API (`__init__.py`, `api.py`, `types.py`)

The package re-exports a flat API from `__init__.py`. The main orchestration lives in `api.py`:

- `compare_one_to_one(query, target, ...)` → `ComparisonResult`
- `compare_one_to_many(query, targets, ...)` → `list[ComparisonResult]`
- `create_null_distribution(models, ...)` → `NullBuildSummary`
- `run_one_to_one(config)`, `run_one_to_many(config)`, `run_null_distribution(config)`

These functions build immutable frozen dataclass configs (`OneToOneConfig`, `OneToManyConfig`, `NullBuildRequest`), resolve model references (path/GenericModel), resolve sequences (FASTA path / random batch), validate strategy/metric compatibility, and dispatch to `comparison.compare` or `nulls.build`.

**Immutable contracts** (`types.py`):
- `ComparatorConfig` — frozen dataclass with 15 fields: `metric`, `seed`, `n_jobs`, `pfm_mode`, `pfm_top_fraction`, `search_range`, `min_logfpr`, `window_radius`, `realign_window`, `profile_normalization`, `cache_mode`, `cache_dir`, `background`, `pvalue`, `null_distribution`, `null_search_dirs`, `effective_number_of_targets`.
- `ComparisonResult` — frozen dataclass with Mapping interface; fields: `query`, `target`, `score`, `offset`, `orientation`, `metric`, `n_sites`, `p_value`, `adj_p_value`, `e_value`, `null_id`, `null_n`, `null_estimator`. JSON key aliases: `p-value`, `adj.p-value`, `E-value`. None-valued significance fields are omitted from serialization.

### 2.2 Model representation (`models.py`, `handlers.py`)

`GenericModel(type_key, name, representation, length, config)` is a mutable dataclass with `eq=False`. It is the single model container for all model families. The `config` dict carries model-specific metadata: `kmer`, `_source_pfm` (PWM), `minimum`/`maximum` (SiteGA), `order` (BaMM), `scores_data` (score profiles).

The `registry: dict[str, ModelHandler]` maps `type_key` → `ModelHandler(scan, scan_both, load, write, score_bounds)`. Handlers are registered at import time via `register_builtin_handlers()` in `handlers.py`. Six type keys: `pwm`, `bamm`, `sitega`, `dimont`, `slim`, `scores`.

**Patterns NOT to port:**
- `GenericModel` with `Any` representation and `dict` config → Julia uses concrete immutable structs per model family.
- `registry` dict with string keys → Julia uses multiple dispatch.
- `config` dict for model-specific parameters → Julia uses typed fields.

### 2.3 Batches (`batches.py`)

Three `TypedDict` containers with `values` + `lengths` + optional `mask`/`padding_value`:

- `SequenceBatch` — int8 padded 2D array, `padding_value=4` (N/padding).
- `MaskedBatch` — float32 2D scores with boolean mask, `padding_value=0.0`.
- `ProfileBundle` — 3D `(2, n_rows, max_width)` float32, one per strand.

Helpers: `make_sequence_batch`, `make_masked_batch`, `make_score_batch`, `make_strand_bundle`, `pack_batch`, `pack_profile_bundle`, `flatten_valid`, `flatten_profile_bundle`, `row_values`, `profile_row_values`, `batch_with_values`.

**Patterns NOT to port:**
- TypedDict with runtime-validated dict shape → Julia uses concrete parametric structs.
- Padded dense arrays with masks as canonical representation → Julia uses ragged arrays (offset-based) as canonical, padded only as a kernel optimization.

### 2.4 Scanning (`scanning.py`, `functions/scanning.py`)

`scanning.py` dispatches scan requests through the handler registry. `StrandMode = Literal["best", "+", "-", "both"]`. The actual Numba kernels in `functions/scanning.py`:

- `batch_all_scores(sequences, matrix, kmer, is_revcomp, with_context)` → `MaskedBatch`
- `batch_all_scores_strands(sequences, matrix, kmer, with_context)` → `(MaskedBatch, MaskedBatch)`
- `score_seq(num_site, kmer, model)` → scalar (standalone, not Numba-compiled)

Encoding: 5-ary, `A=0, C=1, G=2, T=3, N/padding=4`. Context codes computed as `code = code * 5 + encoded_base`. Forward scoring: positions scanned left-to-right. Reverse scoring: positions scanned with reverse-complement lookup using `complement = 3 - base`, reading from the same window but reversed.

Score bounds: `score_bounds_from_representation(matrix)` = `(min over all axes except last, sum)(matrix), (max ..., sum)(matrix)`.

**Scan geometry:** For a model with `kmer` (context size + 1) and `with_context`:
- `context_len = kmer - 1` if `with_context` else `0`
- `window_size = motif_len + context_len`
- `n_terms = window_size - kmer + 1`
- Output positions: `max(length - motif_len + 1, 0)` per sequence.

Bucket optimization: when sequence lengths vary, rows are bucketed by output length (step 32) and scored in contiguous sub-batches to avoid processing padding columns.

**Patterns NOT to port:**
- Numba `@njit(fastmath=True)` with 5-ary encoding and bucketing → Julia uses ordinary loops with explicit bounds; encoding and bucketing decisions deferred to benchmark.
- `fastmath=True` → not copied without benchmark + ADR.

### 2.5 Comparison (`comparison/`)

Two strategies dispatched through `runner.py`'s `registry: dict[str, Callable]`:

**Motif strategy** (`motif.py`): direct matrix alignment.
- Normalizes motif tensor to `[alphabet_axes..., position]` layout, detects 4/5/power-of-4 alphabet axes, reshapes higher-order to 4-ary.
- Reverse complement: transpose alphabet axes in reverse order, flip all axes.
- Aligns by sliding query over target for all offsets; minimum overlap = `min(w1, w2) / 2`.
- Metrics: `pcc` (column Pearson, averaged), `ed` (negative mean Euclidean distance), `cosine` (column cosine, averaged).
- Four orientation candidates: `++`, `+-`, `-+`, `--`. Tie-breaking: `max(score, -orientation_rank)` where rank is `++=0, +-=1, -+=2, --=3`.
- `pfm_mode`: when True or models differ in type, reconstructs PFM from sequences before comparison.

**Profile strategy** (`profile.py`): window-based profile alignment.
- Scans both strands → raw `ProfileBundle`.
- Fits normalizer from background (or foreground if no background): `build_score_log_tail_table` on flattened scores.
- Applies normalizer: `apply_score_log_tail_table_to_profile_bundle` → normalized bundle.
- Prepares bundle: `prepare_profile_bundle` ensures contiguous float32.
- For each orientation pair in `[(++, 0,0), (--, 1,1), (+-, 0,1), (-+, 1,0)]`:
  - Collects anchors: best per row, or threshold-selected positions (`min_logfpr`).
  - For each shift in `[-search_range, search_range]`:
    - Collects candidate windows from both anchor sets with realignment.
    - Scores via fused Numba kernel: `score_shift` in `functions/alignment.py`.
  - Tie-breaking: higher score → more sites → smaller `|shift|`.
- Metrics: `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine`.

**Fused alignment kernel** (`functions/alignment.py`):
- `score_shift` dispatches to `align_shift_serial` or `align_shift_parallel` (Numba `prange`).
- Threshold for parallel: `n_rows * profile_width * (2*search_range+1) >= 100 000`.
- Workspace: `AlignmentWorkspace(marks, positions, partials)` — reusable row-local storage.
- Anchor CSR: `(positions, offsets)` arrays from `build_anchor_csr`.
- Candidate deduplication via `marks[row, position] == generation` (generational marking).
- Reductions: pooled overlap for CO/Dice, rowwise averages for rowwise metrics, cosine accumulation.

**Patterns NOT to port:**
- Strategy registry dict → Julia multiple dispatch.
- Numba `prange` parallelism inside kernels → Julia threads at top level (sequences, targets, pairs).
- `@njit(fastmath=True)` → deferred.
- Numba thread scope (`set_num_threads`) → Julia `ThreadedExecution` policy.
- Runtime cache dicts with `id(batch)` keys → Julia preallocation and explicit reuse.

### 2.6 Sites and PFM reconstruction (`sites.py`)

- `get_sites(model, sequences, mode, fpr_threshold, strand, ...)` → `pd.DataFrame` with columns: `seq_index`, `start`, `end`, `strand`, `score`, `log_tail`, `site`.
- `get_pfm(model, sequences, mode, fpr_threshold, strand, ..., top_fraction, pseudocount)` → `np.ndarray` PFM.
- Selection modes: `best` (one best hit per sequence) or `threshold` (all hits ≥ FPR threshold).
- Hit collection: scans required strands, collects positions/scores, sorts by `(seq_index asc, score desc, start asc, strand_idx asc)`.
- `top_fraction`: keeps top-scoring fraction of hits via `argpartition`.
- Site extraction: numeric windows from encoded sequences; reverse hits complemented (`3 - base`), padding preserved.
- PCM → PFM: `pcm_to_pfm(pcm, pseudocount=0.25)` = `(pcm + 0.25) / (col_sum + 4*0.25)`.
- `log_tail` annotation via threshold table lookup.

**Patterns NOT to port:**
- `pd.DataFrame` as site result → Julia returns typed `SiteCollection` structs; DataFrame via extension.
- `np.add.at` for PCM → Julia uses explicit loops or `counts[code, pos] += 1`.

### 2.7 Null distributions (`nulls/`)

- `build_null_distributions(models, relations, ...)` → `NullBuildResult`.
- Relations: `parse_group_relations(path)` reads TSV/CSV with `motif`/`group` columns; eligible pairs are motifs from different groups.
- For each query, compares against all eligible targets (sorted by name), collects raw scores.
- Fits GEV: `GenextremeSurvivalEstimator(scores)` uses `scipy.stats.genextreme.fit`.
- Annotation: `sf(score)` → p-value, `pvalue * effective_n` → E-value, `stats.false_discovery_control(method="bh")` → adjusted p-values.
- Storage: `joblib.dump`/`joblib.load` — `.joblib` files.
- Metadata: `NULL_FORMAT_VERSION=2`, strategy, metric, sequence/background fingerprints, model collection fingerprint, relation fingerprint.
- Compatibility: `load_compatible_null_distribution_file` searches explicit path + `~/.cache/mimosa/nulls/*.joblib`.

**Patterns NOT to port:**
- `joblib` as storage format → Julia uses versioned portable schema (JSON+binary blobs).
- `scipy.stats.genextreme.fit` → native Julia GEV with parameterization audit (ADR 0005).
- `scipy.stats.false_discovery_control` → native BH implementation.
- `pandas` for relation reading → Julia CSV reader.

### 2.8 Cache (`cache.py`)

- `fingerprint_model(model)` → blake2b hash of `type_key + name + length + kmer + representation bytes`.
- `fingerprint_batch(batch)` → blake2b of values + mask + lengths.
- Cache path: `{cache_dir}/v8/profiles/{profile_kind}/{sequence_fp}/{background_fp}/{model_fp}.npz`.
- Atomic write: `tempfile.mkstemp` → `np.savez` → `os.replace`.
- `load_profile_cache` → `np.load(allow_pickle=False)`, validates version, returns `ProfileBundle`.
- `clear_cache(cache_dir)` → `shutil.rmtree`.

**Patterns NOT to port:**
- `np.savez` cache format → Julia uses versioned schema with checksums.
- `allow_pickle=False` guard → Julia doesn't use unsafe deserialization.
- Global cache dir under `~/.cache` → Julia uses explicit `Cache` object.

### 2.9 I/O (`io/`)

| Format | Reader | Writer | Notes |
|---|---|---|---|
| MEME | `read_meme(path, index)` → `(PFM[4,w], (name, w), count)` | — | Letter-probability matrix, transposed to `[base, position]`. Multi-motif via `read_meme_many`. |
| PFM | `read_pfm(path)` → `(PFM, length)` | `write_pfm(pfm, name, length, path)` | Plain text, auto-detects orientation (4 or 5 rows → transpose). |
| BaMM | `read_bamm(path, order)` → log-odds tensor `[5, ..., 5, length]` | `joblib.dump` | `.ihbcp` format. Log-odds vs uniform background. Ambiguous (N) axis filled with min. |
| SiteGA | `read_sitega(path)` → `(5×5×length, name, length)` | `write_sitega(model, path)` | Dinucleotide weights. 5-ary with N handling. |
| Dimont XML | `read_dimont(path)` → `(tensor, length, span)` | — | Jstacs XML, MarkovModelDiffSM tree, dense 5-ary tensor. |
| Slim XML | `read_slim(path)` → `(tensor, length, span)` | — | Jstacs XML, component/ancestor mixture, log-sum-exp. |
| Scores | `read_scores(path)` → `MaskedBatch` | — | FASTA-like format with float values. |
| FASTA | `read_fasta(path)` → `SequenceBatch` | — | A/C/G/T encoded as 0-3, everything else = 4. |
| DIST | — | `write_dist(table, max, min, path)` | Normalized threshold table. Not used in main pipeline. |

**Security concerns to address in Julia:**
- No size limits on declared dimensions.
- No path traversal checks in ZIP/container (future).
- XML parsed via `xml.etree.ElementTree` without entity restrictions.
- `joblib.load` is unsafe deserialization — only trusted inputs.

### 2.10 CLI (`cli.py`)

Commands: `profile`, `motif`, `build-null`, `cache clear`.

Output: `json.dumps(result.to_dict())` to stdout. Logs to stderr via `TqdmLoggingHandler`.

Exit codes: 0 on success, 1 on validation error or invalid mode. Exceptions propagate with stacktrace (no `--debug` gate currently — a Julia improvement).

Arguments mapped to `ComparatorConfig` via `map_args_to_comparator_kwargs`. `--jobs` controls Numba thread count, not Julia threads.

## 3. Dependency map

| Dependency | Usage | Julia replacement |
|---|---|---|
| NumPy ≥2.0 | Array operations everywhere | Julia `AbstractArray` / concrete arrays |
| SciPy ≥1.14 | `genextreme.fit`, `genextreme.sf/pdf`, `false_discovery_control` | Native GEV + BH |
| pandas ≥2.2 | `get_sites` output, `parse_group_relations` | Typed structs + CSV.jl |
| joblib ≥1.5 | Model/null persistence (pickle) | Versioned portable schema |
| numba ≥0.65 | All numerical kernels (`@njit`, `@njit(parallel=True)`) | Ordinary Julia loops + threads |
| tqdm | Progress bars | ProgressLogging.jl (optional extension) |

## 4. Numerical risk areas

1. **GEV fitting**: `scipy.stats.genextreme.fit` uses MLE with specific initialization, sign convention (`shape` parameter sign), and optimizer. Julia must audit parameterization, not just call a look-alike function.
2. **Float32 accumulation**: Scan kernels use `np.float32` with `fastmath=True`. Profile alignment uses `np.float64` partials but `np.float32` inputs. Potential drift needs measurement.
3. **5-ary encoding**: `code = code * 5 + base` means ambiguous bases participate in scoring. N state scores are filled with `min` over concrete bases — this is a deliberate modeling choice, not a bug.
4. **Empirical log-tail**: `-log10(cumulative_count / total)` with descending sort. Edge cases: empty sample (returns `[[0,0]]`), single unique score, repeated scores.
5. **Tie-breaking**: Orientation priority `++ > +- > -+ > --` is encoded in `ORIENTATION_TIEBREAK` dict and `_select_best_orientation`. Profile shift ties: more sites wins, then smaller `|shift|`.
6. **Reverse complement coordinates**: Reverse scan reads from the same window positions but with complemented/reversed base lookup. This must be frozen as a fixture before implementing Julia scanning.

## 5. Julia correspondence plan

| Python module | Julia module | Notes |
|---|---|---|
| `models.py` + `handlers.py` | `Mimosa.jl/src/models.jl` → `models/` | Concrete types: `PWM`, `PFM`, `BaMM`, `SiteGA`, `Dimont`, `Slim`, `ScoreProfile` |
| `batches.py` | `Mimosa.jl/src/sequences.jl` | `EncodedSequenceBatch`, `RaggedArray` |
| `scanning.py` + `functions/scanning.py` | `Mimosa.jl/src/scanning.jl` → `scanning/` | Dispatch by model type; serial kernels |
| `comparison/` | `Mimosa.jl/src/comparison.jl` → `comparison/` | Metric types, alignment, results |
| `functions/alignment.py` | `Mimosa.jl/src/comparison/alignment.jl` | Serial fused kernel; threading at top level |
| `functions/tails.py` | `Mimosa.jl/src/profiles/normalization.jl` | `EmpiricalLogTail` fit/apply |
| `functions/profile.py` | `Mimosa.jl/src/comparison/metrics.jl` | `PearsonCorrelation`, `EuclideanSimilarity`, etc. |
| `functions/matrices.py` | `Mimosa.jl/src/models/matrices.jl` | `pfm_to_pwm`, `pcm_to_pfm` |
| `sites.py` | `Mimosa.jl/src/sites.jl` → `sites/` | `SiteHit`, `SiteCollection`, selectors |
| `nulls/` | `Mimosa.jl/src/statistics.jl` → `statistics/` | Native GEV, BH, null schema |
| `cache.py` | `Mimosa.jl/src/cache.jl` | Versioned, checksummed, explicit |
| `io/` | `Mimosa.jl/src/io.jl` → `io/` | Safe parsers with size limits |
| `cli.py` | `Mimosa.jl/app/` or `ext/` | Thin adapter |
| `api.py` + `types.py` | `Mimosa.jl/src/Mimosa.jl` | Public API exports |