# Mimosa.jl

A Julia package for motif scanning, comparison, and statistical evaluation.
Independent Julia implementation of the [MIMOSA](https://github.com/mimosa/mimosa)
motif comparison toolkit, redesigned for Julia's multiple dispatch, parametric
types, and column-major layout.

## Features

- **Six model families:** PWM, PFM, BaMM, SiteGA, Dimont, Slim
- **Multiple file formats:** MEME, PFM, BaMM `.ihbcp`, SiteGA `.mat`, Dimont/Slim XML,
  score profiles (FASTA-like)
- **Motif comparison:** Direct matrix/tensor alignment with PCC, Euclidean distance,
  cosine similarity
- **Profile comparison:** Score-profile-based comparison with overlap coefficient,
  dice, and cosine metrics
- **Site extraction:** Best-per-sequence, threshold, and top-fraction selection
- **PFM reconstruction:** From selected sites with pseudocount and orientation correction
- **Null distributions:** Native GEV fitting, BH FDR, E-values, p-value annotation
- **Parallelism:** Serial and threaded execution with deterministic results
- **Portable storage:** Versioned TOML manifest + NPY binary blobs with checksum validation
- **Content-based cache:** Atomic writes, checksum validation, corruption recovery
- **CLI:** Thin adapter over public API, JSON output, stable exit codes
- **Security:** Bounded parsing, path traversal protection, strict NPY validation

## Installation

The package is not yet registered in the General registry. Install from a local
clone:

```julia
using Pkg
Pkg.develop(path="/path/to/Mimosa.jl")
```

Or from the repository (once published):

```julia
using Pkg
Pkg.add(url="https://github.com/mimosa-jl/Mimosa.jl.git")
```

## Quick start

```julia
using Mimosa

# Read a motif from MEME or PFM format (auto-detected by extension).
query = readmodel("examples/pif4.meme")
target = readmodel("examples/gata2.meme")

# Compare two PWMs by direct matrix alignment.
result = compare(query, target; metric=:pcc)
# => ComparisonResult("pwm_model", "MA0036.2", 0.4336f0, -1, "+-", "pcc")

# Serialize to JSON matching the Python CLI schema.
println(to_json(result))

# Read sequences from a FASTA file.
batch, names = readsequences("examples/foreground.fa")

# Scan sequences with strand policies.
scores = scan(query, batch; strands=BestStrand())

# Threaded batch scan (deterministic results).
scores_t = scan(query, batch; strands=BestStrand(), execution=ThreadedExecution(4))

# Build a null distribution from a collection of models.
models = [query, target]
relations = parse_group_relations("groups.tsv")
null_result = build_null(models, relations; strategy="motif", metric=:pcc)
dist = null_result.distribution  # NullDistribution

# Save and load null distributions (portable bundle format).
savenull("null_dist", dist)
loaded_dist = loadnull("null_dist")

# Annotate comparison results with p-values.
annotated = annotate_results([result], dist; effective_number_of_targets=1)

# Write and read models in portable bundle format.
writemodel("output/model_bundle", query)
loaded_model = readmodel("output/model_bundle")
```

## CLI

```bash
# Direct motif comparison
julia --project=Mimosa.jl app/mimosa.jl motif examples/pif4.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm --metric pcc

# Profile-based comparison with random sequences
julia --project=Mimosa.jl app/mimosa.jl profile examples/pif4.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm --metric co --num-sequences 50 --seq-length 100

# Build a null distribution
julia --project=Mimosa.jl app/mimosa.jl build-null examples/ \
  --model-type pwm --groups groups.tsv --strategy motif --output null_dist

# Inspect a model
julia --project=Mimosa.jl app/mimosa.jl inspect-model examples/foxa2.ihbcp --type bamm

# Convert a legacy model to portable bundle format
julia --project=Mimosa.jl app/mimosa.jl convert-model examples/pif4.meme output/pif4_bundle
```

See [CLI documentation](docs/src/cli.md) for all commands and options.

## Architecture

- **Library-first:** the public API is the Julia module; the CLI is a thin
  adapter that parses arguments, calls the API, and serializes output.
- **Concrete domain types:** `PWM`, `PFM`, `BaMM`, `SiteGA`, `Dimont`, `Slim`
  are parametric immutable structs. No `Any` fields, no string dispatch in hot
  paths.
- **Multiple dispatch:** metrics are small types; `compare` dispatches on model
  types and metric types.
- **Type-stable kernels:** all scanning and comparison kernels return concrete
  types with zero per-position allocations.
- **One-based internal indexing:** coordinate conversion to zero-based
  half-open CLI output happens at the serialization boundary.

## Development

```bash
# Run tests
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# Format check
julia --project=Mimosa.jl -e 'using JuliaFormatter; @assert format("Mimosa.jl/src"; overwrite=false); @assert format("Mimosa.jl/test"; overwrite=false)'

# Downstream contract test (separate environment)
julia --project=Mimosa.jl/test/downstream Mimosa.jl/test/downstream/runtests.jl
```

## License

MIT. See [LICENSE](LICENSE).