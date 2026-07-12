# Storage Format

Mimosa.jl uses a **language-neutral bundle** format for model and null
distribution storage: a TOML manifest + binary NPY blobs.

## Design principles

- **No unsafe deserialization**: Never uses `pickle`, `joblib`, or Julia
  `Serialization` for user-facing storage
- **Language-neutral**: NPY binary format is compatible with NumPy
- **Versioned**: Schema version in manifest enables future migrations
- **Checksum-verified**: SHA-256 of all binary blobs
- **Atomic writes**: Complete sibling staging directory + rename; orphan
  staging directories are ignored by readers

## Model bundle structure

```
output_dir/
├── manifest.toml    # Metadata, schema version, checksums
├── weights.npy      # Binary Float32 matrix (row-major for numpy)
└── (model-specific files)
```

### `manifest.toml` fields

| Field | Description |
|-------|-------------|
| `format` | `"mimosa"` magic identifier |
| `format_version` | Positive integer (currently 1) |
| `kind` | `"pwm"`, `"pfm"`, `"bamm"`, `"sitega"`, `"dimont"`, `"slim"` |
| `name` | Model name string |
| `dtype` / `shape` | Top-level array declaration; v1 uses `"<f4"` and `[rows, columns]` |
| `layout` | `"row_major"` |
| `background` | 4-element tuple (PWM only) |
| `order` / `span` | Higher-order model parameter (if applicable) |
| `arrays.<name>` | Relative file, dtype, shape and exact SHA-256 checksum |

### NPY format

Binary blobs use the standard NumPy `.npy` format with:
- Magic: `\x93NUMPY`
- Supported version headers: NPY 1.0 and 2.0
- Little-endian Float32 (`<f4`)
- Row-major (C-contiguous) layout for Python compatibility

Readers reject malformed headers, unsupported dtypes/endianness, rank or shape
mismatches, truncated or extra payload bytes, and non-finite model/null data.
Manifest paths must remain inside the bundle root, and every referenced blob
requires a checksum in the exact form `sha256:<64 lowercase hex>`.

## Null distribution format

```
output_dir/
├── manifest.toml       # Metadata, GEV parameters, schema
├── raw_scores.npy      # Raw comparison scores (Float32)
└── pairs.json           # Contributing pair identifiers
```

### `manifest.toml` fields

| Field | Description |
|-------|-------------|
| `format` | `"mimosa"` |
| `format_version` | Positive integer (currently 1) |
| `kind` | `"null_distribution"` |
| `strategy` | `"motif"` or `"profile"` |
| `metric` | Metric name |
| `n_null` | Number of null comparisons |
| `n_queries` | Number of query models |
| `gev_shape` | GEV shape parameter k |
| `gev_location` | GEV location μ |
| `gev_scale` | GEV scale σ |
| `gev_converged` | Boolean |
| `arrays.raw_null_scores` | Relative file, `<f8` shape and exact SHA-256 checksum |

## Atomic writes

1. Create a sibling staging directory in the target parent
2. Write all blobs and the manifest into the staging directory
3. Flush each completed file
4. Rename the complete staging directory to the target
5. If any step fails, the target is not committed and the staging directory is
   removed best-effort; a process-termination orphan is ignored by readers

## Legacy format conversion

Legacy Python `pickle`/`joblib` models are converted via separate Python
scripts (`scripts/convert_legacy_model.py`, `scripts/convert_legacy_null.py`)
with an explicit `--trusted-input` security guard. The Julia package never
reads `pickle` or `joblib` files.
