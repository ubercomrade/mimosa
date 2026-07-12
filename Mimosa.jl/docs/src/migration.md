# Python Migration

## Overview

Mimosa.jl is an independent Julia implementation of the Python MIMOSA
project. It preserves scientific semantics and user-facing capabilities while
using idiomatic Julia architecture.

## What changed

### Architecture

| Python | Julia |
|--------|------|
| `GenericModel(type_key, representation, config)` | Concrete immutable structs per family |
| `registry: Dict[str, ModelHandler]` | Multiple dispatch on concrete types |
| `TypedDict` batches with `values`/`mask`/`lengths` | `RaggedArray` and `EncodedSequenceBatch` |
| `pandas.DataFrame` for sites/relations | Typed `SiteCollection`, `GroupRelations` |
| Numba bucketing, `fastmath=True`, thread-mask scope | Serial kernels + top-level execution policy |
| `joblib`/`pickle` storage | TOML manifest + NPY blobs |
| `scipy.stats.genextreme.fit` | Native BFGS MLE fit |
| String type keys in hot paths | Concrete parametric types, no string dispatch |

### What was not ported

- Python-specific Numba `fastmath=True` and bucketing patterns
- pandas/DataFrames in internal kernels (typed structs instead)
- joblib/pickle as user-facing storage format
- Global mutable state of any kind

## Converting legacy models

### Step 1: Convert models

```bash
# Trust the input explicitly (pickle may contain arbitrary objects)
python scripts/convert_legacy_model.py \
    --trusted-input old_model.pkl \
    --output new_model_bundle
```

### Step 2: Convert null distributions

```bash
python scripts/convert_legacy_null.py \
    --trusted-input old_null.joblib \
    --output new_null_dir
```

### Step 3: Verify in Julia

```julia
using Mimosa

# Load converted model
model = readmodel("new_model_bundle")
println("Type: ", typeof(model))
println("Width: ", length(model))
println("Score bounds: ", scorebounds(model))

# Verify scan works
batch = readsequences("sequences.fa")
scores = scan(model, batch; strands=BestStrand())
```

## Compatibility fixtures

The `tests/fixtures/compatibility/` directory contains frozen Python oracle
outputs for 89+ fixtures across all model families. These verify that Julia
produces results within documented tolerances.

## Known numerical differences

See [Numerical Compatibility](@ref) for the full tolerance classes and known
divergences. Summary:

- Raw scan scores: Float32 with Float64 cross-column accumulation, `atol ≤ 1e-5`
- GEV parameters: different optimizer, `atol=0.01`, `rtol=0.05`
- Random sequences: different RNG, not bit-compatible
- Integer values (offsets, indices, counts): exact match