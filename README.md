# Mimosa.jl

A Julia package for motif scanning, comparison, and statistical evaluation,
migrating the Python [MIMOSA](../) implementation to an independent Julia
package.

## Features

- **Multiple model families:** PWM, PFM, BaMM, SiteGA, Dimont, Slim
- **Multiple file formats:** MEME, PFM, BaMM `.ihbcp`, SiteGA `.mat`, Dimont/Slim XML
- **Motif comparison:** Direct matrix alignment with PCC, Euclidean distance, and cosine similarity
- **Profile comparison:** Score-profile-based comparison with overlap, dice, and cosine metrics
- **Site extraction:** Best-per-sequence, threshold, and top-fraction selection
- **PFM reconstruction:** From selected sites with pseudocount and orientation correction
- **Null distributions:** Native GEV fitting, BH FDR, E-values
- **Parallelism:** Serial and threaded execution with deterministic results
- **Portable storage:** Versioned TOML manifest + NPY binary blobs
- **Content-based cache:** Atomic writes, checksum validation, corruption recovery
- **CLI:** Thin adapter over public API, JSON output, stable exit codes

## Quick start

```julia
using Mimosa

# Read a motif from MEME or PFM format.
query = readmodel("examples/pif4.meme")
target = readmodel("examples/gata2.meme")

# Compare two PWMs by direct matrix alignment.
result = compare(query, target; metric=:pcc)
# => ComparisonResult("pwm_model", "MA0036.2", 0.4336f0, -1, "+-", "pcc")

# Serialize to JSON matching the Python CLI schema.
println(to_json(result))
```

## CLI

```bash
julia --project=Mimosa.jl app/mimosa.jl motif \
  --query examples/pif4.meme --target examples/gata2.meme --metric pcc
```

Commands: `motif`, `profile`, `build-null`, `cache clear`, `inspect-model`, `convert-model`.

## Architecture

- **Library-first:** the public API is the Julia module; the CLI is a thin
  adapter that parses arguments, calls the API, and serializes output.
- **Concrete domain types:** `PFM`, `PWM`, `BaMM`, `SiteGA`, `Dimont`, `Slim`
  are parametric immutable structs per [ADR 0001](../docs/adr/0001-model-type-hierarchy.md).
  No `Any` fields, no string dispatch.
- **Multiple dispatch:** metrics, strand policies, execution policies are small
  types; `scan`, `compare`, `scorebounds` dispatch on model types.
- **Type-stable hot kernels:** all scanning kernels return concrete types with
  zero per-position allocations.
- **One-based internal indexing:** coordinate conversion to zero-based
  half-open CLI output happens at the serialization boundary per
  [ADR 0006](../docs/adr/0006-coordinate-offset-orientation-conventions.md).

## Development

```bash
# Install and test
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate()'
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'

# Format code
julia --project=Mimosa.jl -e 'using JuliaFormatter; format(".", BlueStyle())'

# Run benchmarks
julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/benchmarks.jl

# Build documentation
julia --project=Mimosa.jl/docs Mimosa.jl/docs/make.jl
```

## Documentation

- [Quick Start](Mimosa.jl/docs/src/quickstart.md)
- [Julia API](Mimosa.jl/docs/src/api.md)
- [CLI](Mimosa.jl/docs/src/cli.md)
- [Supported Models](Mimosa.jl/docs/src/models.md)
- [Data Layout](Mimosa.jl/docs/src/data_layout.md)
- [Numerical Compatibility](Mimosa.jl/docs/src/numerical_compatibility.md)
- [Reproducibility](Mimosa.jl/docs/src/reproducibility.md)
- [Storage Format](Mimosa.jl/docs/src/storage.md)
- [Security](Mimosa.jl/docs/src/security.md)
- [Python Migration](Mimosa.jl/docs/src/migration.md)
- [Extending Mimosa](Mimosa.jl/docs/src/extending.md)
- [MotifHORDE Contract](Mimosa.jl/docs/src/downstream_contract.md)
- [Architecture](Mimosa.jl/docs/src/architecture.md)

## License

MIT. See [LICENSE](LICENSE).