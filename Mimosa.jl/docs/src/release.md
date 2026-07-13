# Release Candidate

This page documents the platform support matrix, clean-room validation
procedures, rollback policy, and migration window for Mimosa.jl release
candidate.

## Platform support matrix

| Platform | Julia version | Status | CI coverage |
|----------|---------------|--------|-------------|
| Linux x86_64 | 1.10 (minimum) | **Supported** | `ubuntu-latest`, 1/4 threads |
| Linux x86_64 | 1.12.6 (latest stable) | **Supported** | `ubuntu-latest`, 1/4 threads |
| macOS arm64 | 1.12.6 (latest stable) | **Supported** | `macos-latest`, 4 threads |
| macOS x86_64 | 1.10+ | Expected to work | Not separately tested in CI |
| Windows x86_64 | 1.10+ | **Experimental** | Not tested in CI |
| All platforms | nightly | Allowed failure | `continue-on-error: true` |

### Support policy

- **Minimum supported Julia version**: 1.10 (LTS). All compat bounds are set to
  `julia = "1.10"`. Stdlib compat bounds (`Printf`, `Random`, `SHA`, `TOML`)
  are set to `"1.10"` to avoid requiring a newer Julia.
- **Latest stable**: tested on every push and pull request.
- **Nightly**: tested in CI with `continue-on-error: true`. Breakages on nightly
  do not block pull requests but are tracked for future support decisions.
- **Windows**: not officially supported. The package uses only pure-Julia
  dependencies and standard file I/O, so it is expected to work, but no CI
  job validates Windows. Users who need Windows support should verify
  independently and report issues.

## Clean-room installation validation

### Fresh user depot

A fresh depot install verifies that no stale precompiled artifacts or cached
packages are required:

```bash
# Create a temporary fresh depot and install from scratch
JULIA_DEPOT_PATH=/tmp/fresh-depot julia --project=Mimosa.jl -e \
    'using Pkg; Pkg.instantiate(); Pkg.precompile(); using Mimosa; println("OK")'
```

This command must succeed without manual `Pkg.resolve()` and without any Python
runtime. The entire dependency tree is pure Julia.

### Read-only working tree

Tests must pass when the working tree is a clean `git clone`:

```bash
git clone https://github.com/mimosa-jl/Mimosa.jl.git /tmp/mimosa-readonly
cd /tmp/mimosa-readonly
julia --project=Mimosa.jl -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.test()'
```

No test should write to the package source directory. All temporary files are
created via `mktempdir()` in the system temp directory.

### No local absolute paths

The following must contain no hardcoded local paths:

- `Project.toml` — no absolute paths in any field
- `Manifest.toml` — if present, no absolute paths (checked: zero matches)
- `src/` — no hardcoded paths in source code (checked: zero matches)
- `docs/src/` — no hardcoded paths in documentation (checked: zero matches)

Verified by:

```bash
grep -rn "/home/\|/Users/" Mimosa.jl/src/ Mimosa.jl/Project.toml Mimosa.jl/docs/src/
# Expected: no output
```

### Import side effects

`using Mimosa` must NOT:

- Create files or directories on disk
- Print output to stdout or stderr
- Start background threads (beyond Julia's default thread pool)
- Modify global settings (e.g., `BLAS.set_num_threads`, environment variables)

Verified procedure:

```julia
# In a fresh Julia session:
using Mimosa
# No output should appear. Check:
# - No files in working directory
# - No unexpected output
# - Threads.nthreads() reflects only the JULIA_NUM_THREADS setting
```

The `Mimosa` module does not execute any top-level side effects beyond module
definition and precompilation hooks (which run only during precompilation, not
at `using` time).

## CLI artifact validation

The CLI (`app/mimosa.jl`) works without Python installed:

```bash
# Help and version — no model files needed
julia --project=Mimosa.jl app/mimosa.jl --help
julia --project=Mimosa.jl app/mimosa.jl --version

# Model inspection — needs only a model file
julia --project=Mimosa.jl app/mimosa.jl inspect-model examples/pif4.meme

# Motif comparison — needs only model files
julia --project=Mimosa.jl app/mimosa.jl motif examples/pif4.meme examples/pif4.meme \
    --model1-type pwm --model2-type pwm

# Cache management — no dependencies needed
julia --project=Mimosa.jl app/mimosa.jl cache clear

# Model conversion — needs only a model file
julia --project=Mimosa.jl app/mimosa.jl convert-model examples/pif4.meme /tmp/bundle
```

All commands produce JSON to stdout and diagnostics to stderr, with stable exit
codes (0 = success, 1 = usage error, 2 = runtime error). No Python runtime,
`pip`, `conda`, or Python packages are required.

## Rollback policy

Rollback means installing a previous version of the Mimosa.jl package:

```julia
# Install a specific registered version
using Pkg
Pkg.add(name="Mimosa", version="0.0.9")

# Or develop from a specific git tag
Pkg.add(url="https://github.com/mimosa-jl/Mimosa.jl.git", rev="v0.0.9")
```

Previous package versions remain installable from the General registry. The
portable storage format is versioned (`format_version` in `manifest.toml`), so
bundles written by older versions remain readable by newer versions. Bundles
written by newer versions are rejected by older versions with a typed
`ModelFormatError` if the format version is unsupported.

## Migration window

### Dual-support period

The Python MIMOSA implementation remains available and pinned during the
migration window. Both implementations can be used in parallel:

- **Python oracle**: the existing Python MIMOSA package remains the scientific
  oracle. Frozen Python fixtures in `tests/fixtures/compatibility/` are the
  source of truth for numerical compatibility.
- **Julia implementation**: Mimosa.jl is the target implementation. All new
  features and bug fixes are developed in Julia.
- **Duration**: dual-support continues until RC review is complete and
  downstream consumers (MotifHORDE.jl) have accepted the Julia API.
- **Owner**: Mimosa contributors are responsible for compatibility fixes during
  this period.

### Compatibility fix commitment

- Frozen Python fixtures are never regenerated without separate scientific
  review. Any new numerical divergence between Julia and Python requires:
  1. A documented rationale (ADR if the divergence is intentional)
  2. A regression test that asserts the new behavior
  3. An entry in [Numerical Compatibility](@ref) documenting the divergence class
- The Julia implementation does not use `fastmath` or other unsafe
  floating-point optimizations. All numerical differences arise from
  legitimate sources (different RNG, different optimizer, Float32
  accumulation) documented in the tolerance classes.

### Deprecation timeline

1. **Current phase (dual-support)**: Both Python and Julia are maintained.
   Users are encouraged to migrate to the Julia API.
2. **After RC review and downstream acceptance**: Python implementation is
   declared legacy/deprecated. No new features are added to Python. Critical
   bug fixes only.
3. **Python code removal**: NOT part of this remediation plan. A separate
   decision will be made after the migration window ends.

### Migration steps for users

#### Step 1: Install Julia

Install Julia 1.10 or later from [julialang.org](https://julialang.org/) or via
[juliaup](https://github.com/JuliaLang/juliaup):

```bash
# Via juliaup
curl -fsSL https://install.julialang.org | sh
juliaup add 1.10
julia +1.10 --version
```

#### Step 2: Add Mimosa.jl

```julia
# From the General registry (once registered)
using Pkg
Pkg.add("Mimosa")

# Or develop from a local clone
Pkg.develop(path="path/to/Mimosa.jl")
```

#### Step 3: Convert legacy models

Convert Python pickle/joblib models to the portable TOML+NPY bundle format:

```bash
# Convert a single model
julia --project=Mimosa.jl app/mimosa.jl convert-model old_model.pkl new_bundle

# Or via the Julia API
julia --project=Mimosa.jl -e '
using Mimosa
model = readmodel("old_model.meme")
writemodel(model, "new_bundle")
'
```

See [Python Migration](@ref) for details on converting null distributions and
verifying results.

#### Step 4: Verify against Python oracle

Run compatibility tests to confirm Julia results match Python within documented
tolerances:

```bash
julia --project=Mimosa.jl -e 'using Pkg; Pkg.test()'
```

The compatibility test suite covers 89+ fixtures across all model families.
See [Numerical Compatibility](@ref) for tolerance classes and known divergences.

#### Step 5: Integrate Julia API

Replace Python MIMOSA calls with Julia equivalents:

```julia
using Mimosa

# Read models
model = readmodel("model_bundle")

# Scan sequences
batch = readsequences("sequences.fa")
scores = scan(model, batch; strands=BothStrands())

# Compare motifs
result = compare(query_model, target_model, sequences; metric=:co)

# Build null distribution
result = build_null(models, relations; sequences=sequences, metric=:co)
null = result.distribution

# Annotate results with p-values
annotated = annotate_results(results, null; pvalue=0.05)
```

See [Quick Start](@ref) and the `Julia API` page for the full API reference.

## PackageCompiler and static binary

PackageCompiler.jl can be used to create a standalone system image for faster
startup, but this is not required for library usage. A static binary app is
considered separately and does not block the library release candidate.

```julia
# Optional: create a system image with Mimosa precompiled
using PackageCompiler
create_sysimage(:Mimosa; sysimage_path="mimosa_sys.so")
```

## Conda/Bioconda strategy

Mimosa.jl is a pure-Julia package with no Python runtime dependency. Conda or
Bioconda packaging would wrap the Julia installation, not add Python as a
dependency. This is deferred until downstream demand is confirmed and does not
block the release candidate.
