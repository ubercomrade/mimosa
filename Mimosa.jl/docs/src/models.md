# Supported Models

Mimosa.jl supports six model families, each with a concrete immutable type
and specific file format.

## Model types

| Type | Family | Format | Description |
|------|--------|--------|-------------|
| `PWM{T,M,B}` | Matrix | MEME | Position Weight Matrix with N-state row |
| `PFM{T,M}` | Matrix | PFM | Position Frequency Matrix |
| `BaMM{T,M}` | Higher-order | `.ihbcp` | Bayesian Markov Model |
| `SiteGA{T,M}` | Higher-order | `.mat` | Dinucleotide model |
| `Dimont{T,M}` | Higher-order | XML | Jstacs Bayesian network |
| `Slim{T,M}` | Higher-order | XML | Jstacs GenDisMix classifier |
| `ScoreProfile` | Pseudo-model | FASTA-like | Precomputed score profiles |

## File formats

### MEME (PWM)

Standard MEME format. The parser reads the log-odds matrix, converts to 5-row
extended PWM (A, C, G, T, N), with the N row as per-column minimum.

### PFM (Position Frequency Matrix)

`PFM` is a matrix representation and is not directly scannable. Convert it
explicitly with `pwm_from_pfm(pfm)` before calling `scan`; this keeps the
background and pseudocount choice visible at the API boundary.

`ScoreProfile` is a precomputed profile, not a motif. Its `length` is the
number of profile rows for compatibility; use `nrows(profile.scores)` for
that value. It intentionally has no `motif_length` or `window_size`.

Simple whitespace-separated frequency matrix with 4 rows (A, C, G, T).

### BaMM `.ihbcp`

Bayesian Markov Model with higher-order context. The `.ihbcp` format stores
a flattened matrix of shape `(5^(order+1), motif_length)`.

### SiteGA `.mat`

Dinucleotide model with 25-row representation (5×5 dinucleotide codes),
flattened from Python's `(5, 5, length)` to `(25, length)`.

### Dimont XML

Jstacs XML format. The parser extracts `MarkovModelDiffSM` parameters,
materializes them into a dense 5-ary tensor, and flattens to
`(5^(span+1), motif_length)`.

### Slim XML

Jstacs XML format. The parser extracts `SLIM` component/ancestor parameters,
normalizes via log-sum-exp, and materializes into the same representation
as Dimont.

## Score bounds

Each model type implements `scorebounds(model)` returning `(min_score, max_score)`
—the per-column minimum and maximum, summed across positions. This is used
for score normalization and null distribution fitting.

## Scanning

All models support the same scanning API via multiple dispatch:

```julia
scan(model, sequence; strands=BestStrand())
scan(model, batch; strands=BestStrand(), execution=SerialExecution())
scan!(dest, model, sequence; strands=ForwardOnly())
```

The scanning kernel for higher-order models (BaMM, Dimont, Slim) uses a shared
generic implementation in `_ho_scan_forward!` / `_ho_scan_reverse!` parameterized
by geometry (kmer, context, window, n_terms). SiteGA uses a dinucleotide-specific
kernel. PWM uses a direct 5-row matrix lookup.

## Type parameters

All matrix model types are parametric:

```julia
PWM{T<:AbstractFloat, M<:AbstractMatrix{T}, B<:NTuple{4,AbstractFloat}}
BaMM{T<:AbstractFloat, M<:AbstractMatrix{T}}
```

This allows the compiler to specialize on the concrete element type (typically
`Float32`) and array type (typically `Matrix{Float32}`), ensuring type-stable
hot kernels with zero per-position allocations.
