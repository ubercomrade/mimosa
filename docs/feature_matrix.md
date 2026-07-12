# Feature Matrix

> **Stage 0 audit artifact.** Records every user-facing capability of the Python implementation
> and its planned status in `Mimosa.jl`. Updated at each gate.

## Legend

| Status | Meaning |
|---|---|
| planned | Julia will support this |
| in-progress | Julia implementation underway |
| done | Julia implementation complete and compatibility-verified |
| deferred | Post-1.0 or downstream (`MotifHORDE.jl`) |
| not-porting | Deliberately not carried to Julia |

## Model families

| Model | Python type_key | Parser | Writer | Scanning | Sites | Reconstruction | Julia status |
|---|---|---|---|---|---|---|---|
| PWM | `pwm` | MEME, PFM, pickle | PFM | forward/reverse/best/both | yes | PFM | MEME/PFM parser done (Stage 1); scanning planned |
| BaMM | `bamm` | `.ihbcp`, pickle | joblib | forward/reverse/best/both + context | yes | PFM | planned |
| SiteGA | `sitega` | `.mat`, pickle | `.mat` | forward/reverse/best/both | yes | PFM | planned |
| Dimont | `dimont` | XML, pickle | joblib | forward/reverse/best/both + context | yes | PFM | planned |
| Slim | `slim` | XML, pickle | joblib | forward/reverse/best/both + context | yes | PFM | XML parser + scanning done (Stage 5d) |
| Score profiles | `scores` | FASTA-like | — (not writable) | identity (no scan) | — | — | planned |

## Comparison strategies

| Strategy | Python entry | Metrics | Julia status |
|---|---|---|---|
| Motif (direct matrix alignment) | `strategy_motif` | `pcc`, `ed`, `cosine` | done (Stage 1) |
| Profile (window-based) | `strategy_profile` | `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine` | planned |
| One-to-one | `compare_one_to_one` | all of the above | planned |
| One-to-many | `compare_one_to_many` | all of the above | planned |

## Strand policies

| Policy | Python `StrandMode` | Behavior | Julia status |
|---|---|---|---|
| Forward only | `"+"` | Scan forward strand only | planned |
| Reverse only | `"-"` | Scan reverse complement only | planned |
| Best | `"best"` | Per-position max of forward/reverse | planned (default) |
| Both | `"both"` | Return both strand tracks separately | planned |

## Orientations

| Orientation | Python label | Meaning | Tie-break rank |
|---|---|---|---|
| Forward-Forward | `++` | Query forward, target forward | 0 (highest priority) |
| Forward-Reverse | `+-` | Query forward, target reverse | 1 |
| Reverse-Forward | `-+` | Query reverse, target forward | 2 |
| Reverse-Reverse | `--` | Query reverse, target reverse | 3 |

Tie-breaking: when scores are equal, the lower rank wins. Within the same orientation, offset ties are broken by traversal order (negative to positive). Profile shift ties: more sites wins, then smaller `|shift|`.

## Normalization

| Method | Python name | Description | Julia status |
|---|---|---|---|
| Empirical log-tail | `empirical_log_tail` | Sort scores, compute cumulative tail probability, `-log10(tail)`, descending lookup table | planned (sole method) |

## Site selection

| Mode | Python `mode` | Description | Julia status |
|---|---|---|---|
| Best per sequence | `best` | One highest-scoring hit per sequence | planned |
| Threshold | `threshold` | All hits at or above FPR-derived score threshold | planned |
| Top fraction | (via `top_fraction` param) | Keep top-scoring fraction of hits for PFM reconstruction | planned |

## Null distributions

| Feature | Python implementation | Julia status |
|---|---|---|
| Group relations | `parse_group_relations` (pandas CSV/TSV) | planned (CSV.jl) |
| Eligible pair scheduling | sorted by name, cross-group only | planned |
| Score collection | `compare_one_to_many` per query | planned |
| GEV fitting | `scipy.stats.genextreme.fit` | planned (native, ADR 0005) |
| P-value (upper tail) | `scipy.stats.genextreme.sf` | done (`pvalue`, `annotate_results`) |
| E-value | `pvalue * effective_number_of_targets` | done (`annotate_results`; CLI override) |
| BH FDR | `scipy.stats.false_discovery_control(method="bh")` | done (native Benjamini-Hochberg) |
| Storage | `joblib.dump` (pickle) | done (hardened TOML + NPY bundle; bounded, checksummed, staged writes) |
| Compatibility check | metadata fingerprint comparison | done for CLI annotation (strategy, metric, sequence/background fingerprints) |
| Auto-search | `~/.cache/mimosa/nulls/*.joblib` | deferred (explicit bundle path only; no hidden global search) |

## Cache

| Feature | Python implementation | Julia status |
|---|---|---|
| Profile cache | `.npz` files, `CACHE_VERSION="v8"` | planned (versioned, checksummed) |
| Model fingerprint | blake2b of type_key + name + length + kmer + representation | planned (content-based) |
| Batch fingerprint | blake2b of values + mask + lengths | planned |
| Atomic write | tempfile → replace | planned |
| Clear | `shutil.rmtree` | planned |
| Global cache dir | `~/.cache/mimosa/nulls/` | not-porting (explicit object only) |

## CLI commands

| Command | Python subcommand | Key arguments | Julia status |
|---|---|---|---|
| Profile comparison | `mimosa profile` | model1, model2, --model1-type, --model2-type, --metric, --fasta, --background, --search-range, --window-radius, --realign-window, --min-logfpr, --pvalue, --null-distribution, --effective-number-of-targets | partial (workflow cache deferred) |
| Motif comparison | `mimosa motif` | model1, model2, --model1-type, --model2-type, --metric, --pfm-mode, --pfm-top-fraction, --pvalue, --null-distribution, --effective-number-of-targets | done |
| Build null | `mimosa build-null` | motifs, --model-type, --groups, --strategy, --metric, --output, --fasta, --background | done |
| Cache clear | `mimosa cache clear` | --cache-dir | done |
| (new) Inspect model | — | — | planned |
| (new) Convert model | — | — | planned |
| (new) Convert null | — | — | planned |

## I/O formats

| Format | Read | Write | Julia status |
|---|---|---|---|
| MEME | yes (single + multi) | — | done (Stage 1) |
| PFM | yes | yes | reader done (Stage 1); writer planned |
| BaMM `.ihbcp` | yes | joblib only | planned (new writer) |
| SiteGA `.mat` | yes | yes | planned |
| Dimont XML | yes | joblib only | planned (new writer) |
| Slim XML | yes | joblib only | parser done (Stage 5d); writer planned |
| Score FASTA | yes | — | planned |
| DNA FASTA | yes | — | planned |
| DIST | — | yes | deferred (not in main pipeline) |
| joblib/pickle | yes (trusted) | yes | not-porting (converter only) |

## Python patterns NOT ported

| Pattern | Why not | Julia replacement |
|---|---|---|
| `GenericModel` with `Any` representation | No type stability | Concrete immutable structs per model family |
| `registry: dict[str, ModelHandler]` | String dispatch | Multiple dispatch |
| `TypedDict` batches with dict shape | Runtime validation only | Concrete parametric structs |
| Numba `@njit(fastmath=True)` | Unsafe without benchmark proof | Ordinary loops; unsafe annotations only after profiling |
| Numba `prange` inside kernels | Non-composable parallelism | Top-level thread scheduling |
| `pd.DataFrame` as core return type | Heavy dependency | Typed structs; DataFrame via extension |
| `joblib`/pickle storage | Unsafe deserialization | Versioned JSON+binary schema |
| `scipy.stats` direct calls | Parameterization risk | Native GEV with audit |
| Global mutable cache directory | Hidden side effects | Explicit `Cache` object |
| `id(batch)` in runtime cache keys | Session-dependent | Preallocation and explicit reuse |
| `os.makedirs` on import | Import-time side effect | No I/O on import |
| `set_num_threads` scope | Numba-specific | `ThreadedExecution` policy |

## Test coverage (Python baseline)

| Test file | Tests | Coverage area |
|---|---|---|
| `tests/unit_io_models.py` | — | Parsers, model loading, registry |
| `tests/unit_comparison.py` | — | Motif and profile comparison, metrics |
| `tests/unit_functions.py` | — | Scanning, tails, matrices, curves |
| `tests/unit_collections_nulls.py` | — | Null distributions, relations, GEV |
| `tests/test_integration.py` | ~30 | CLI subprocess tests for all commands |
| **Total declared** | **146** | |

## Public API surface (Python `__init__.py`)

```
ComparatorConfig, ComparisonResult, NullBuildRequest, NullBuildSummary,
OneToManyConfig, OneToOneConfig, GenericModel, StrandMode,
clear_cache, compare, compare_one_to_one, compare_one_to_many,
create_comparator_config, create_null_distribution, create_null_distribution_config,
create_one_to_many_config, create_one_to_one_config,
get_frequencies, get_pfm, get_scores, get_sites,
build_null_distributions, load_null_distribution_file, parse_group_relations,
read_model, read_models, register_model_handler,
run_null_distribution, run_one_to_one, run_one_to_many,
scan_model, save_null_distribution_file, validate_metric
```
