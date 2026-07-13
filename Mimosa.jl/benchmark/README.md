# Mimosa.jl Benchmark Suite

Reproducible benchmark suite for Mimosa.jl, covering all representative workloads
from `PLAN.md` and `PLAN_2.md` E1/E2.

## Quick start

```bash
# Run the full benchmark suite (prints human-readable + JSON to stdout)
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl

# Save results to a file
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --output results.json

# Print environment metadata only
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --report

# Compare against stored baseline
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl --baseline baseline.json
```

## Thread configuration

Set `JULIA_NUM_THREADS` before launching Julia to control thread count:

```bash
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl
JULIA_NUM_THREADS=1 julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl
```

## Options

| Option | Description |
|--------|-------------|
| `--output <file>` | Write JSON results to file (default: stdout) |
| `--report` | Print environment metadata only, no benchmarks |
| `--baseline <file>` | Compare results against stored baseline |
| `--samples <n>` | BenchmarkTools samples per benchmark (default: 100) |
| `--seconds <f>` | BenchmarkTools seconds budget per benchmark (default: 5.0) |
| `--help`, `-h` | Show help |

## Workloads covered

### PWM scanning
- Single-sequence forward, best-strand, and reverse complement
- Widths: 8, 15, 30
- Sequence lengths: 100, 200, 1000

### Batch scanning (ragged heavy-tail)
- Small (100 × 200), medium (1000 × 200), large (10000 × 200)
- Variable-length ragged batches (short-heavy, long-heavy)
- Serial and threaded scaling at 1/2/4 threads

### Motif comparison (direct matrix alignment)
- All metrics: PCC, Euclidean distance, cosine similarity
- Width combinations: 8×8, 8×15, 15×30, 8×30

### One-to-many profile comparison
- Target counts: 10, 100, 1000
- CO metric with search range and window radius

### Higher-order scanning (BaMM)
- Orders 0–5 (synthetic models)
- Real BaMM models from `examples/` (foxa2, gata2, gata4, myog — all order 4)

### Site extraction and PFM reconstruction
- BestPerSequence (low site density)
- ThresholdHits (high site density)
- TopFractionHits (medium density)
- Batch sizes: 100, 1000

### GEV fitting and statistics
- Sample sizes: 100, 500, 2000
- BH FDR correction (1000 p-values)

### Null distribution building
- Motif strategy (PCC and ED metrics)
- Profile strategy (CO metric)
- Serial and threaded execution

### Storage (bundle write/read)
- PWM model round-trip
- BaMM model round-trip

### Startup and import latency
- `using Mimosa` in fresh subprocess
- `Pkg.precompile()` time
- CLI subprocess startup (`mimosa --version`)

## Metrics collected

Each benchmark records:

| Metric | Description |
|--------|-------------|
| `median_ns` | Median time in nanoseconds |
| `min_ns` | Minimum time in nanoseconds |
| `mean_ns` | Mean time in nanoseconds |
| `variance_ns` | Variance of time samples |
| `allocations` | Number of allocations |
| `memory_bytes` | Total bytes allocated |
| `n_samples` | Number of samples taken |
| `n_evals` | Number of evaluations per sample |
| `warmup` | Whether warm-up was performed (always true) |
| `parameters` | Workload-specific parameters (width, order, n_seqs, etc.) |

## Environment metadata

The report includes:

- Git commit SHA
- Julia version and executable path
- Machine architecture, OS, kernel
- CPU model, speed, cores
- Thread count
- Total and free RAM
- Package versions (Mimosa and dependencies)
- Timestamp
- Warm-up policy
- Sample count and seconds budget

## Regression baseline (E2)

The `baseline.json` file stores per-benchmark median timings for regression
comparison. Per `PLAN_2.md` E2:

- The baseline should be stored for a **controlled machine** or compared using
  stable normalized metrics.
- Scheduled CI publishes a comparison report but **does NOT block PRs** on noisy
  microbenchmarks.
- RC gate blocks only on **confirmed regressions** of agreed representative
  workloads.
- Any optimization change must include profile evidence and a compatibility rerun.

### Updating the baseline

```bash
# On a controlled machine with stable configuration:
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl \
  --output Mimosa.jl/benchmark/baseline.json
```

### Comparing against baseline

```bash
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/runbenchmarks.jl \
  --baseline Mimosa.jl/benchmark/baseline.json
```

Results with a ratio > 1.25× are flagged as potential regressions.

## Legacy benchmark file

The older `benchmarks.jl` file (Stage 9) is kept for backwards compatibility
but `runbenchmarks.jl` supersedes it with full PLAN_2.md E1/E2 coverage
including machine-readable JSON output, environment metadata, and baseline
comparison.