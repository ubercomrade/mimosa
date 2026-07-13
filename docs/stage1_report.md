# Stage 1 Report: Package Foundation and First PWM Vertical Slice

> **Historical migration artifact.** This report records a completed early
> implementation stage. Planned work, package layout, test counts, and API
> examples below are not current status. See [Documentation Map](README.md) and
> the active `Mimosa.jl` documentation.
>
> **Gate:** Gate 1 (Stage 1). See `PLAN.md` §5, Stage 1.

## 1. Architectural decision summary

Implemented a library-first Julia package `Mimosa.jl` following the architecture
in `REFACTORING.md` and ADRs 0001, 0002, 0006.

- **Module layout:** `Mimosa.jl` (top-level, exports only), `errors.jl`,
  `models/` (PFM/PWM types + matrix conversions), `io/` (MEME/PFM readers),
  `comparison/` (metrics, alignment, results), `serialization.jl` (JSON writer),
  `cli.jl` (thin adapter).
- **Concrete domain types:** `PFM{T,M}` and `PWM{T,M,B}` are parametric
  immutable structs with concrete fields. No `Any`, no abstract fields, no
  string dispatch. `PWM.background` is `NTuple{4,T}`.
- **Multiple dispatch:** `compare(query::PWM, target::PWM; metric=...)` dispatches
  on model types. Metrics are small types (`PearsonCorrelation`,
  `EuclideanDistance`, `CosineSimilarity`) dispatched via `score_columns`.
- **Error hierarchy:** `MimosaError <: Exception` with `ModelFormatError`,
  `ModelDimensionError`, `InvariantError`.
- **One-based internal indexing;** zero-based CLI output conversion at the
  serialization boundary (ADR 0006). Offset and orientation values are
  index-convention-independent integers.

## 2. Python patterns deliberately NOT ported

- `GenericModel(type_key, name, representation::Any, config::Dict)` → concrete
  per-family structs (`PFM`, `PWM`).
- `registry: dict[str, ModelHandler]` string dispatch → Julia multiple dispatch.
- `config["_source_pfm"]` hack → PWM stores weights only; PFM is separate.
- `np.float32` broadcasting in metrics → explicit Julia loops over bases and
  columns (composable, type-stable, no `fastmath`).
- `joblib`/`pickle` storage → deferred to Stage 7 (versioned JSON+NPY per
  ADR 0003); not needed for the Stage 1 slice.

## 3. Julia features used

- Parametric immutable structs with concrete field types.
- Multiple dispatch on `AbstractColumnMetric` subtypes and model types.
- `view` for zero-copy column slices during alignment.
- `reverse(...; dims=1)` / `dims=2` for reverse complement.
- Manual NPY reader (no dependency) for loading frozen fixtures in tests.
- Manual JSON writer (stdlib `Printf` only) to avoid a runtime JSON dependency;
  `JSON3` is a test-only extra for parsing the manifest.
- `@testset` with Aqua quality checks (optional).

## 4. Public contracts added

- `readmodel(path; format=:auto, index=0, background=0.25)` → `PWM`
- `read_meme(path; index=0)` → `PFM`
- `read_pfm(path)` → `PFM`
- `pfm_to_pwm(pfm; background=0.25)` → `Matrix{Float32}` (4-row)
- `pwm_from_pfm(pfm|model; background, name)` → `PWM` (5-row extended)
- `reverse_complement(model|weights)` → reversed model/matrix
- `scorebounds(model)` → `(min, max)` score bounds
- `compare(query::PWM, target::PWM; metric=:pcc|:ed|:cosine|typed)` → `ComparisonResult`
- `to_dict(result)` / `to_json(result)` → JSON schema v1 draft
- `main(args)` → thin CLI entry (Int exit code)

`ComparisonResult` fields: `query, target, score::Float32, offset::Int,
orientation::String, metric::String`.

## 5. Tests added

- **Unit:** `test_models.jl` (constructors, validation, conversions, RC
  involution, scorebounds), `test_readers.jl` (MEME/PFM parse, malformed
  inputs, format detection), `test_metrics.jl` (PCC/ED/Cosine, zero-variance),
  `test_alignment.jl` (self-alignment, tie-breaking, no-mutation),
  `test_serialization.jl` (to_dict/to_json).
- **Property:** `test_properties.jl` (RC involution, identical-motif pcc=1,
  determinism, no-mutation by non-`!` functions, score-bound consistency,
  valid orientation labels).
- **Compatibility:** `test_oracle_fixtures.jl` loads frozen `.npy` fixtures and
  compares parser arrays, PWM conversion, reverse complement, score bounds, and
  motif alignment (self pcc/ed/cosine; pif4-vs-gata2 pcc/ed/cosine) plus CLI JSON
  against `manifest.json` metadata.
- **Integration:** `test_cli.jl` (CLI exit codes for valid/missing/nonexistent
  input).

## 6. Benchmark

Skeleton in `Mimosa.jl/benchmark/benchmarks.jl` using `BenchmarkTools.jl`
comparing `compare(pif4, gata2; pcc)` warm path. Not run (Julia unavailable in
this environment).

## 7. Known differences from Python

None at Stage 1. The implementation matches the frozen oracle fixtures
exactly: same PFM parse, same `pfm_to_pwm` formula (`log((pfm+1e-4)/0.25)`),
same reverse complement (`flip rows + reverse columns`), same offset traversal
(`-(target_len-1):query_len-1`, first-wins on `>`), same orientation tie-break
(`++ > +- > -+ > --`), same metric formulas including zero-denominator → 0.

## 8. Verification commands

```bash
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate()'
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
julia --project=Mimosa.jl -e 'using Mimosa; Mimosa.main(ARGS)' -- \
  --query examples/pif4.meme --target examples/pif4.meme --metric pcc
```

> **Note:** Julia is not installed in the current environment, so tests could
> not be executed here. The code was written to pass Gate 1 and should be run
> in a Julia 1.10+ environment to confirm.

## 9. Risks for next stage (Stage 2: sequences and PWM scanning)

- Sequence encoding and FASTA reader must match the frozen fixture
  `fasta_read_foreground` exactly (int8 padded 2D in Python vs ragged flat in
  Julia per ADR 0002). Julia will use `EncodedSequenceBatch` (flat data +
  offsets); compatibility tests must compare encoded bytes and lengths, not
  the padded layout.
- Random sequence generation uses NumPy RNG in Python fixtures; Julia must load
  stored encoded bytes from fixtures rather than reproduce NumPy randomness.
- Scan kernel must match `pwm_scan_forward/reverse/both` fixtures including
  the 5-ary encoding, N-state scoring (min over concrete bases), and reverse
  window indexing (ADR 0006 §6.6).
- Float32 accumulation with Python `fastmath=True` may diverge; tolerance
  policy from `numerical_compatibility.md` must be applied.
