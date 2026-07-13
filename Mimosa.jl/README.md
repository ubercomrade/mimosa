# Mimosa.jl

Mimosa.jl is a Julia 1.10+ package for DNA motif scanning, profile-based
comparison, site extraction, PFM reconstruction, and statistical evaluation.
It supports PWM/PFM, BaMM, SiteGA, Dimont, Slim, and precomputed score profiles.

Comparison is profile-only. Motif models are scanned against an explicit
`EncodedSequenceBatch`, normalized by empirical score tails, anchored, and
aligned across strands. Available metrics are `co`, `co_rowwise`, `dice`,
`dice_rowwise`, and `cosine`.

## Installation

The package is not currently registered in General. Install it from a local
clone:

```julia
using Pkg
Pkg.develop(path="/path/to/mimosa/Mimosa.jl")
```

## Quick Start

From the repository root:

```julia
using Mimosa

query = readmodel("examples/pif4.meme")
target = readmodel("examples/gata2.meme")
sequences, _ = readsequences("examples/foreground.fa")

result = compare(query, target, sequences; metric=:co)
println(to_json(result))

scores = scan(query, sequences; strands=BestStrand())
sites = selectsites(query, sequences, BestPerSequence())
pfm = reconstruct_pfm(query, sequences, BestPerSequence())
```

Threaded execution must be enabled both in the Julia runtime and in the API:

```julia
scores = scan(
    query,
    sequences;
    strands=BestStrand(),
    execution=ThreadedExecution(4),
)
```

Start that process with `julia --threads=4` or `JULIA_NUM_THREADS=4`.

## CLI

```bash
JULIA_NUM_THREADS=4 julia --project=Mimosa.jl Mimosa.jl/app/mimosa.jl \
  profile examples/pif4.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm --fasta examples/foreground.fa \
  --metric co --threads 4
```

Current commands are `profile`, `build-null`, `cache clear`, `inspect-model`,
and `convert-model`. The removed direct matrix API and `motif` command are not
available.

## Design

- Concrete immutable model and execution-policy types drive multiple dispatch.
- Sequence and score batches use flat offset-based storage.
- Serial execution is the default; threaded work is bounded and preserves
  input order.
- User-facing storage uses bounded TOML manifests plus checksum-verified NPY
  blobs. Bundle writes are atomic.
- The CLI writes JSON only to stdout and diagnostics only to stderr.

## Development

Run commands from the repository root:

```bash
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

julia --project=Mimosa.jl/test -e \
  'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'

julia --project=Mimosa.jl/docs Mimosa.jl/docs/make.jl

JULIA_NUM_THREADS=4 julia --project=Mimosa.jl/benchmark \
  Mimosa.jl/benchmark/runbenchmarks.jl
```

See the [quick start](docs/src/quickstart.md), [CLI guide](docs/src/cli.md),
[API reference](docs/src/api.md), and [architecture](docs/src/architecture.md).

## License

MIT. See [LICENSE](LICENSE).
