# Storage Format

Mimosa.jl stores models and null distributions as directory bundles containing
a bounded TOML manifest and checksum-verified NPY blobs. It never uses pickle,
joblib, or Julia `Serialization` for user-facing input.

## Model Bundles

Model bundles use format version 1:

```text
model_bundle/
├── manifest.toml
└── data/
    └── representation.npy
```

The manifest records kind, name, dtype, shape, layout, model-specific geometry,
relative blob path, and SHA-256 checksum. Supported kinds are `pwm`, `pfm`,
`bamm`, `sitega`, `dimont`, and `slim`.

Use `writemodel(path, model)` and `readmodel(path)`.

## Null Bundles

Null bundles use format version 2:

```text
null_bundle/
├── manifest.toml
└── data/
    └── raw_null_scores.npy
```

The strategy is always `"profile"`. The manifest stores the profile metric,
Float64 GEV metadata, comparison counts, skipped queries, compatibility
fingerprints, and the checksum/shape declaration for raw scores.

Use `savenull(path, distribution)` and `loadnull(path)`.

## NPY and Layout

In-memory matrices follow Julia column-major conventions. NPY blobs are
explicitly row-major for NumPy interoperability. Readers support only reviewed
little-endian Float32/Float64 shapes used by current writers.

## Security and Atomicity

Readers reject traversal, absolute or escaping paths, symlink escape, malformed
NPY headers, unsupported dtype/version/rank, shape or payload mismatch,
non-finite values, invalid checksums, and oversized declarations before
allocation.

Writers build a complete sibling staging directory and commit it by rename.
Failed writes do not replace valid targets; orphan stages are ignored.

Any schema change requires a version increment, compatibility/migration checks,
tests, and updated documentation.
