# Storage and Cache

## Model bundles

`write_model(path, model)` writes a directory bundle for any built-in model:

```text
model_bundle/
├── manifest.toml
└── data/
    ├── weights.bin
    └── representation.bin
```

Only the model-specific payload is present: PWM uses `weights.bin`; higher-order
models and SiteGA use `representation.bin`. The manifest records the model
kind, name, shape, geometry, row-major layout, payload length, and SHA-256
checksum.

Model payloads are raw little-endian Float32 values with no NPY header. Bundle
writes use a staging directory and atomic rename. Existing targets are not
overwritten.

## Null bundles

Null distributions use a versioned directory format:

```text
null_bundle/
├── manifest.toml
└── data/
    ├── raw_null_scores.npy
    └── pair_indices.npy
```

Raw scores are Float64 NPY data. Pair indices are stored in manifest-label
index order and validated against the label table on read. The manifest stores
the metric, model family, seed, sampling version, comparison contract, and
payload checksums.

Use the I/O utilities from `mimosa.io`:

```python
from mimosa.io import read_null_bundle, write_null_bundle

write_null_bundle("output/null_bundle", distribution)
stored = read_null_bundle("output/null_bundle")
```

## Prepared-profile cache

The optional cache stores prepared profiles in a versioned binary payload with
separate score, offset, and anchor sections:

```python
from mimosa.cache import Cache
from mimosa import prepare_profile

prepared = prepare_profile(
    model,
    sequences,
    cache=Cache(".mimosa-cache"),
)
```

Disk hits verify the payload checksum once per cache instance and expose the
numeric sections through read-only memory maps. The cache does not retain
prepared profiles in a process-local store, so each disk hit maps the prepared
sections from the cache entry. Entries from incompatible cache formats are
treated as misses; Mimosa never loads pickle cache payloads.

Prepared-profile writes stream aligned sections directly into one staged binary
payload while calculating SHA-256 incrementally. This avoids holding a second
full payload in Python memory. The benchmark's `--phase-timings` mode reports
cache encode, lock wait, write, checksum, and semantic-validation timings.

Cache keys include the model or score-profile fingerprint, sequence and
background fingerprints, the Float32 `min_logerr` bit pattern, normalization
settings, and algorithm versions. A change to any score-affecting input creates
a cache miss. Clear entries with:

```bash
uv run mimosa cache clear --cache-dir .mimosa-cache
```
