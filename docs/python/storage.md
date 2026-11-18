# Storage and Cache

## Model bundles

`write_model(path, model)` writes a directory bundle for an enabled built-in
model (PWM, BaMM, Dimont or Slim):

```text
model_bundle/
├── manifest.toml
└── data/
    ├── weights.bin
    └── representation.bin
```

Only the model-specific payload is present: PWM uses `weights.bin`; higher-order
models use `representation.bin`. SiteGA bundles remain on disk but cannot
currently be read or written. The manifest records the model
kind, name, shape, geometry, row-major layout, payload length, and SHA-256
checksum.

Model payloads are raw little-endian Float32 values with no NPY header. Bundle
writes use a staging directory and atomic rename. Existing targets are not
overwritten.

## Monte Carlo results

Significance is evaluated per pair with direct Monte Carlo; no shared null
distribution needs to be built or stored. The old null-bundle readers and
writers are no longer available. Save the returned arrays yourself, or save
the CLI JSON output, which includes Monte Carlo settings and exceedance counts.
See [migration notes](migration.md).

## Prepared-profile cache

The optional cache stores threshold-independent exact normalized profiles in a
versioned binary payload with separate score and offset sections. The current
prepared-profile format is v8, with a stored circular-layout flag; anchors are
derived for the requested threshold rather than serialized:

```python
from mimosa.cache import Cache
from mimosa import prepare_profile

prepared = prepare_profile(
    model,
    sequences,
    min_logerr=2.0,
    circular=True,
    cache=Cache(".mimosa-cache"),
)
```

Disk hits verify each unchanged payload checksum once per cache instance and expose the
numeric sections through read-only memory maps. The cache does not retain
prepared profiles in a process-local store, so each disk hit maps the prepared
sections from the cache entry and derives anchors for the requested
`min_logerr`. Entries from incompatible cache formats are treated as misses;
Mimosa never loads pickle cache payloads.
Checksums detect accidental corruption, not malicious tampering; do not load
untrusted cache directories.

Prepared-profile writes stream aligned sections directly into one staged binary
payload while calculating SHA-256 incrementally. This avoids holding a second
full payload in Python memory. The benchmark's `--phase-timings` mode reports
cache encode, lock wait, write, checksum, and semantic-validation timings.

Cache keys include the model or score-profile fingerprint, sequence and
background fingerprints, normalization settings, the linear or circular layout,
and algorithm versions. Linear and circular profiles of the same model are
separate entries.
Changing `min_logerr` reuses the same normalized scores and rebuilds only the
anchors. A change to any score-affecting input creates a cache miss. Clear
entries with:

```bash
uv run mimosa cache clear --cache-dir .mimosa-cache
```

Clearing removes recognized checksum-valid cache entries, including older
entries whose metadata and checksum still validate, but preserves unrelated
files and checksum-invalid entries.
It is not an unconditional directory deletion.

## Monte Carlo working storage

The public `direct_mc` function stores fixed target profiles in host memory or
isolated temporary `.npy` files (`allow_pickle=False`). Its `storage`,
`memory_limit`, `temp_dir` and `prefetch` settings govern that working storage,
not the prepared-profile cache. Replicate query profiles are kept in bounded
device batches; target anchors have a separate retention budget.

The CLI additionally supports persistent caching with `--cache-dir`. Combined
with `--pvalue`, it reuses normalized observed query/target profiles from the
cache instead of writing another temporary target copy. Randomized profiles
are not persisted. The Python `direct_mc` function has no public `cache`
parameter.
