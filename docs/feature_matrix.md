# Feature Matrix

This matrix describes the active Julia package in `Mimosa.jl`. Historical
Python behavior is not a supported runtime surface.

| Area | Current support | Public entry points | Notes |
|---|---|---|---|
| Model families | PWM, PFM, BaMM, SiteGA, Dimont, Slim | `readmodel`, concrete model constructors | `ScoreProfile` is a comparison input, not a scannable motif model |
| Scientific formats | MEME, PFM, BaMM `.ihbcp`, SiteGA `.mat`, Dimont/Slim XML | `read_meme`, `read_pfm`, `read_bamm`, `read_sitega`, `read_dimont`, `read_slim` | Format is auto-detected by `readmodel` where possible |
| Sequences | FASTA and generated DNA | `readsequences`, `make_random_sequences`, `EncodedSequenceBatch` | Flat validated `UInt8` storage with `A=0`, `C=1`, `G=2`, `T=3`, ambiguous=`4` |
| Scanning | Forward, reverse, best, both strands | `scan`, `scan!`, `StrandPolicy` | Float32 output in flat `RaggedArray` storage |
| Comparison | Profile-only scalar and one-to-many | `compare`, `prepare_profile`, `PreparedProfile` | Model comparison requires an `EncodedSequenceBatch` |
| Metrics | CO, row-wise CO, Dice, row-wise Dice, cosine | `OverlapCoefficient`, `DiceSimilarity`, `CosineSimilarityProfile` | Stable names: `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine` |
| Site extraction | Best, threshold, top fraction | `selectsites`, `BestPerSequence`, `ThresholdHits`, `TopFractionHits` | One-based inclusive site ranges internally |
| PFM reconstruction | Orientation-aware reconstruction | `reconstruct_pfm`, `build_pcm` | Supports explicit execution policy |
| Statistics | Native GEV, p-values, BH FDR, E-values | `fit_gev`, `pvalue`, `adjusted_pvalues`, `evalue` | GEV calculations use Float64 |
| Null distributions | Profile nulls | `build_null`, `savenull`, `loadnull`, `annotate_results` | Null format version 2; strategy is always `"profile"` |
| Parallel execution | Bounded deterministic task parallelism | `SerialExecution`, `ThreadedExecution` | Runtime threads and API policy are both required |
| Model storage | Portable directory bundles | `writemodel`, `readmodel` | Format version 1, TOML + checksum-verified NPY |
| Cache | Explicit content-addressed cache | `Cache`, `cache_get`, `cache_set`, `clearcache` | Format version 1; no global cache singleton |
| CLI | Five command workflows | `main`, `Mimosa.jl/app/mimosa.jl` | `profile`, `build-null`, `cache clear`, `inspect-model`, `convert-model` |
| Security | Bounded parsing and atomic writes | model/null readers and writers | Rejects traversal, symlink escape, malformed NPY, non-finite data, and oversized declarations |

## Removed Interfaces

The following are deliberately unsupported and must not be restored from old
documentation:

- direct motif matrix/tensor comparison;
- PCC and Euclidean motif metrics;
- the `motif` CLI command;
- the `"motif"` null strategy and null bundles older than version 2;
- unsafe Python pickle/joblib or Julia `Serialization` input.

## Deferred Work

GPU/distributed execution, ZIP bundles, empirical fallback for failed GEV fits,
and automatic cache integration into scientific workflows are not current
public contracts. Any addition requires tests, documentation, and benchmark or
security evidence appropriate to the change.
