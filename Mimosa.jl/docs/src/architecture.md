# Architecture

## Design principles

1. **Library-first**: The public API is the Julia module; the CLI is a thin adapter
2. **Concrete domain types**: `PWM`, `PFM`, `BaMM`, etc. are parametric immutable
   structs — no `GenericModel` with `Any` fields
3. **Multiple dispatch**: Metrics, strand policies, execution policies are small
   types; `scan`, `compare`, `scorebounds` dispatch on model types
4. **Composition over hierarchy**: Independent aspects (strand policy, metric,
   execution) are separate types, not deep inheritance
5. **No string dispatch in hot paths**: String identifiers exist only at I/O/CLI
   boundary, converted to concrete types immediately
6. **Type stability**: All hot kernels return concrete types, zero per-position
   allocations
7. **Serial + top-level parallelism**: Inner kernels are serial and composable;
   parallelism is at the top level (sequences, targets, pairs)

## Module structure

```
src/
├── Mimosa.jl           # Module entry point, exports
├── errors.jl           # Error hierarchy
├── precompile.jl       # PrecompileTools workload
├── serialization.jl    # JSON serialization (to_json, to_dict)
├── cli.jl              # Thin CLI adapter
├── models/             # Model types and constructors
│   ├── types.jl        # Shared model hierarchy, PFM, PWM
│   ├── pwm.jl          # PWM scanning geometry traits
│   ├── bamm.jl         # BaMM type
│   ├── sitega.jl       # SiteGA type
│   ├── dimont.jl       # Dimont type
│   ├── slim.jl         # Slim type
├── sequences/          # Sequence representation
│   ├── encoding.jl     # EncodedSequenceBatch, reverse_complement
│   ├── fasta.jl        # FASTA reader
│   └── ragged.jl       # RaggedArray
├── scanning/           # Scanning interface, validation, and kernels
│   ├── strands.jl       # Strand policies
│   ├── pwm_scan.jl     # Checked PWM boundary and kernels
│   └── higher_order_scan.jl  # Shared higher-order kernels and adapter
├── comparison/         # Profile comparison
│   ├── results.jl      # ComparisonResult, compare()
│   └── profile_comparison.jl  # Profile alignment
├── profiles/           # Profile inputs, normalization, anchors, and alignment
│   ├── precomputed_profile.jl # ScoreProfile precomputed profile source
│   ├── normalization.jl # EmpiricalLogTail
│   ├── anchors.jl       # Anchor collection
│   ├── metrics.jl       # Profile metrics
│   └── alignment.jl    # Shift-based window alignment
├── sites/              # Site extraction and PFM reconstruction
│   └── sites.jl
├── statistics/         # Null distributions and statistics
│   ├── gev.jl          # Native GEV fit
│   ├── pvalues.jl      # p-value, BH FDR, E-value
│   ├── relations.jl    # Group relations
│   ├── null_distribution.jl  # NullDistribution, build_null
│   └── null_storage.jl # savenull, loadnull
├── io/                 # File format parsers
│   ├── motif_readers.jl # MEME, PFM parsers
│   ├── bamm_reader.jl  # BaMM .ihbcp parser
│   ├── sitega_reader.jl # SiteGA .mat parser
│   ├── xml_parser.jl   # Minimal XML parser
│   ├── dimont_reader.jl # Dimont XML parser
│   ├── slim_reader.jl  # Slim XML parser
│   ├── score_reader.jl # Score profile reader
│   └── model_storage.jl # Portable bundle format
├── parallel/          # Execution policies
│   └── parallel.jl
└── cache/              # Content-based cache
    └── cache.jl
```

## ADRs

Architectural decisions are documented in the project's `docs/adr/` directory:

- ADR 0001: Model type hierarchy
- ADR 0002: Sequence representation
- ADR 0003: Storage format
- ADR 0004: Parallelism and RNG
- ADR 0005: GEV fitting
- ADR 0006: Coordinate/offset/orientation conventions

## Type stability guarantees

All hot kernels are type-stable with concrete return types:

| Function | Return type | Allocations |
|----------|-------------|-------------|
| `scan_forward!` | `Vector{Float32}` | 0 |
| `scan_reverse!` | `Vector{Float32}` | 0 |
| `scan_best!` | `Vector{Float32}` | 0 |
| `_ho_scan_forward!` | `Vector{Float32}` | 0 |
| `_ho_scan_reverse!` | `Vector{Float32}` | 0 |
| `reverse_complement!` | `Vector{UInt8}` | 0 |
| `compare` | `ComparisonResult` | ~29 (alignment views) |
| `fit_gev` | `GEVFit` / `GEVFitFailure` | ~120 (BFGS optimizer) |

## Precompilation

`PrecompileTools` workload exercises representative paths during package
precompilation (not at `using Mimosa` time):
- PWM construction and scanning
- Motif comparison (all metrics)
- Site extraction and PFM reconstruction
- GEV fitting
- JSON serialization
- Higher-order model scanning
- Cache fingerprint computation
