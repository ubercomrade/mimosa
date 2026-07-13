# Mimosa.jl

A Julia package for motif scanning, comparison, and statistical evaluation.

Mimosa.jl is an independent Julia implementation of the [MIMOSA](https://github.com/mimosa/mimosa)
motif comparison toolkit, redesigned for Julia's multiple dispatch, parametric types,
and column-major layout.

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

## Package overview

```@docs
Mimosa
```

## Quick navigation

- [Quick Start](quickstart.md): Installation and first steps
- [Julia API](api.md): Public API reference
- [CLI](cli.md): Command-line interface
- [Supported Models](models.md): Model types and file formats
- [Feature Matrix](feature_matrix.md): Comprehensive capability inventory
- [Data Layout](data_layout.md): Matrix layout and coordinate conventions
- [Numerical Compatibility](numerical_compatibility.md): Tolerance classes and known divergences
- [Reproducibility](reproducibility.md): RNG, determinism, and cross-language notes
- [Storage Format](storage.md): Portable model and null distribution format
- [Security](security.md): Safe parsing and untrusted input handling
- [Python Migration](migration.md): Converting legacy models
- [Extending Mimosa](extending.md): Adding new model families
- [MotifHORDE Contract](downstream_contract.md): Downstream API stability guarantee
- [Architecture](architecture.md): Internal design and ADRs
- [Release](release.md): Platform support, validation, and migration window