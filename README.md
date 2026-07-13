# Mimosa.jl

Mimosa.jl is a Julia package for DNA motif scanning, profile-based comparison,
site extraction, PFM reconstruction, and statistical evaluation. The package
is library-first: the command-line interface delegates to the same public API
and emits machine-readable JSON.

The active implementation is in [`Mimosa.jl/`](Mimosa.jl/). The former Python
package is no longer part of the source tree; historical migration documents
are retained only as background.

## Features

- **Model families:** PWM, PFM, BaMM, SiteGA, Dimont, Slim, and precomputed
  `ScoreProfile` inputs
- **Input formats:** MEME, PFM, BaMM `.ihbcp`, SiteGA `.mat`, Dimont/Slim XML,
  FASTA, and FASTA-like score profiles
- **Profile-only comparison:** scan, empirical normalization, anchor
  collection, and strand-aware profile alignment
- **Metrics:** `co`, `co_rowwise`, `dice`, `dice_rowwise`, and `cosine`
- **Sites and PFMs:** best-per-sequence, threshold, and top-fraction site
  selection with orientation-aware PFM reconstruction
- **Statistics:** native GEV fitting, p-values, Benjamini-Hochberg FDR, E-values,
  and profile null distributions
- **Parallelism:** explicit serial or bounded threaded execution with stable
  result order
- **Storage:** checksum-verified TOML manifests and NPY blobs with atomic writes
- **CLI:** JSON on stdout, diagnostics on stderr, and stable exit codes

Direct motif-matrix alignment, PCC/Euclidean motif metrics, the `motif` CLI
command, and the `"motif"` null strategy have been removed.

## Installation

Mimosa.jl requires Julia 1.10 or newer. From a local clone:

```julia
using Pkg
Pkg.develop(path="/path/to/mimosa/Mimosa.jl")
```

For repository development, instantiate the package and test environments:

```bash
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate()'
julia --project=Mimosa.jl/test -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

Run this example from the repository root:

```julia
using Mimosa

query = readmodel("examples/pif4.meme")
target = readmodel("examples/gata2.meme")
sequences, names = readsequences("examples/foreground.fa")

result = compare(query, target, sequences; metric=:co)
println(to_json(result))

scores = scan(query, sequences; strands=BestStrand())
sites = selectsites(query, sequences, BestPerSequence())
pfm = reconstruct_pfm(query, sequences, BestPerSequence())
```

Model-to-model comparison requires an `EncodedSequenceBatch`. For repeated
profile comparisons, prepare the query once:

```julia
query_profile = ScoreProfile("query", scores)
prepared = prepare_profile(query_profile)
results = compare(prepared, [query_profile]; metric=:cosine)
```

Threading is opt-in at both levels: Julia must have runtime threads and the API
must receive `ThreadedExecution`:

```bash
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl -e '
using Mimosa
model = readmodel("examples/pif4.meme")
batch, _ = readsequences("examples/foreground.fa")
scan(model, batch; execution=ThreadedExecution(4))
'
```

## CLI

```bash
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl Mimosa.jl/app/mimosa.jl \
  profile examples/pif4.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm --fasta examples/foreground.fa \
  --metric co --threads 4
```

Current commands are `profile`, `build-null`, `cache clear`, `inspect-model`,
and `convert-model`. `--threads=N` cannot create Julia runtime threads and is
rejected when `N > Threads.nthreads()`.

## Development

```bash
# Full package suite
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

# BlueStyle formatting check
julia --project=Mimosa.jl/test -e \
  'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'

# Documentation
julia --project=Mimosa.jl/docs Mimosa.jl/docs/make.jl

# Benchmarks
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/benchmark \
  Mimosa.jl/benchmark/runbenchmarks.jl
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/benchmark \
  Mimosa.jl/benchmark/bench_1v50.jl
```

## Documentation

- [Documentation map and historical records](docs/README.md)
- [Quick start](Mimosa.jl/docs/src/quickstart.md)
- [Public API](Mimosa.jl/docs/src/api.md)
- [CLI](Mimosa.jl/docs/src/cli.md)
- [Supported models](Mimosa.jl/docs/src/models.md)
- [Data layout](Mimosa.jl/docs/src/data_layout.md)
- [Numerical compatibility](Mimosa.jl/docs/src/numerical_compatibility.md)
- [Reproducibility](Mimosa.jl/docs/src/reproducibility.md)
- [Storage and security](Mimosa.jl/docs/src/storage.md)
- [Architecture](Mimosa.jl/docs/src/architecture.md)

## License

MIT. See [LICENSE](LICENSE).
