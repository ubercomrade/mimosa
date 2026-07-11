# Mimosa.jl

A Julia package for motif scanning, comparison, and statistical evaluation,
migrating the Python [MIMOSA](../) implementation to an independent Julia
package.

> **Stage 1 (current):** PWM/PFM parsing, matrix metrics, motif alignment,
> and typed results. Scanning, profile comparison, null distributions, and CLI
> commands are planned for later stages (see `PLAN.md`).

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
julia -e 'using Mimosa; Mimosa.main(ARGS)' -- \
  --query examples/pif4.meme --target examples/pif4.meme --metric pcc
```

## Architecture

- **Library-first:** the public API is the Julia module; the CLI is a thin
  adapter that parses arguments, calls the API, and serializes output.
- **Concrete domain types:** `PFM` and `PWM` are parametric immutable structs
  per [ADR 0001](../docs/adr/0001-model-type-hierarchy.md). No `Any` fields,
  no string dispatch.
- **Multiple dispatch:** metrics (`PearsonCorrelation`, `EuclideanDistance`,
  `CosineSimilarity`) are small types; `compare` dispatches on model types.
- **One-based internal indexing:** coordinate conversion to zero-based
  half-open CLI output happens at the serialization boundary per
  [ADR 0006](../docs/adr/0006-coordinate-offset-orientation-conventions.md).

## Development

```bash
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate()'
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
```

## License

MIT. See [LICENSE](LICENSE).