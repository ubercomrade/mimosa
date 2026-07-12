# MotifHORDE Downstream Contract

Mimosa.jl is designed as a stable lower layer for the future `MotifHORDE.jl`
orchestration package. This document specifies the API contract that
MotifHORDE.jl can rely on.

## Ownership boundary

### Mimosa.jl owns

- Model types and constructors
- Model readers/writers (all formats)
- Sequence encoding and FASTA I/O
- Scanning (all model families, all strand policies)
- Score calibration and normalization
- Site extraction and PFM reconstruction
- Motif comparison (direct matrix alignment)
- Profile comparison (score-profile-based)
- Null distributions (GEV fit, p-values, FDR, E-values)
- Cache and portable storage
- CLI adapter

### MotifHORDE.jl will own

- Running discovery tools (MEME, BaMM, etc.)
- Parameter grids and search spaces
- Odd/even validation
- Model selection criteria
- Full-data rerun orchestration
- Output directory layout
- Pipeline composition

## Contract: stable public API

The following functions form the stable contract. Their signatures will not
change in breaking ways within a minor version. New keyword arguments may be
added with defaults.

### Model I/O

```julia
readmodel(path; format=:auto, kwargs...) -> AbstractMotifModel
writemodel(path, model) -> Nothing
readsequences(path; kwargs...) -> EncodedSequenceBatch
```

### Scanning

```julia
scan(model, sequence; strands=BestStrand()) -> Vector{Float32} or StrandPair
scan(model, batch; strands=BestStrand(), execution=SerialExecution()) -> RaggedArray or StrandPair
scan!(dest, model, sequence; strands=ForwardOnly()) -> dest
scorebounds(model) -> Tuple{Float32, Float32}
```

### Comparison

```julia
compare(query::AbstractMotifModel, target::AbstractMotifModel; metric=:pcc) -> ComparisonResult
compare(query::ScoreProfile, target::ScoreProfile; metric=..., kwargs...) -> ComparisonResult
```

### Sites and PFM

```julia
selectsites(model, batch, selector; strands=BestStrand()) -> SiteCollection
reconstruct_pfm(model, batch, selector; pseudocount=1e-4f0) -> PFM
```

### Statistics

```julia
build_null(models, relations; execution=SerialExecution(), kwargs...) -> NullDistribution
pvalue(dist::NullDistribution, score::Real) -> Float64
adjusted_pvalues(pvalues; method=BenjaminiHochberg()) -> Vector{Float64}
evalue(pval::Real, n::Int) -> Float64
annotate_results(results, dist; kwargs...) -> Vector{AnnotatedResult}
savenull(path, dist) -> Nothing
loadnull(path) -> NullDistribution
```

### Execution

```julia
SerialExecution() -> SerialExecution
ThreadedExecution(ntasks::Int) -> ThreadedExecution
ThreadedExecution() -> ThreadedExecution  # uses Threads.nthreads()
```

### Cache

```julia
Cache(directory; enabled=true) -> Cache
cache_key(cache, algorithm, parts...) -> String
cache_has(cache, key) -> Bool
cache_get(cache, key) -> Union{Nothing, Vector{UInt8}}
cache_set(cache, key, data) -> Nothing
clearcache(cache) -> Nothing
clearcache(cache, key) -> Nothing
```

## What MotifHORDE should NOT do

- Import internal submodules of Mimosa (e.g., `Mimosa.Scanning`)
- Rely on internal function names (prefixed with `_`)
- Access mutable global state (there is none)
- Modify Mimosa model structs (they are immutable)
- Define methods for Mimosa's abstract types (type piracy)

## Contract test

The `test/downstream/` directory contains a downstream contract test package
that imports only the documented public API and verifies:
- All exported functions are callable
- Types are constructible
- Results have the expected structure
- No access to internals is needed for typical workflows