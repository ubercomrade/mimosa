# Storage Format

Mimosa.jl uses a **language-neutral bundle** format for model and null
distribution storage: a TOML manifest + binary NPY blobs.

## Design principles

- **No unsafe deserialization**: Never uses `pickle`, `joblib`, or Julia
  `Serialization` for user-facing storage
- **Language-neutral**: NPY binary format is compatible with NumPy
- **Versioned**: Schema version in manifest enables future migrations
- **Checksum-verified**: SHA-256 of all binary blobs
- **Atomic writes**: Temp files + rename, no partial files

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
| `format` | `"mimosa-model"` magic identifier |
| `schema_version` | Integer (currently 1) |
| `model_type` | `"pwm"`, `"pfm"`, `"bamm"`, `"sitega"`, `"dimont"`, `"slim"` |
| `name` | Model name string |
| `motif_length` | Number of columns |
| `background` | 4-element tuple (PWM only) |
| `order` / `span` | Higher-order model parameter (if applicable) |
| `weights_shape` | Matrix shape `[rows, cols]` |
| `weights_dtype` | `"<f4"` (little-endian Float32) |
| `weights_checksum` | SHA-256 of `weights.npy` |

### NPY format

Binary blobs use the standard NumPy `.npy` format with:
- Magic: `\x93NUMPY`
- Version header
- Little-endian Float32 (`<f4`)
- Row-major (C-contiguous) layout for Python compatibility

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
| `format` | `"mimosa-null"` |
| `schema_version` | Integer (currently 1) |
| `strategy` | `"motif"` or `"profile"` |
| `metric` | Metric name |
| `n_null` | Number of null comparisons |
| `n_queries` | Number of query models |
| `gev_shape` | GEV shape parameter k |
| `gev_location` | GEV location μ |
| `gev_scale` | GEV scale σ |
| `gev_converged` | Boolean |
| `raw_scores_checksum` | SHA-256 of `raw_scores.npy` |

## Atomic writes

1. Write data to `tempfile` in target directory
2. Write manifest to `manifest.toml.tmp`
3. `fsync` both files
4. `rename` atomically to final names
5. If any step fails, temp files are cleaned up

## Legacy format conversion

Legacy Python `pickle`/`joblib` models are converted via separate Python
scripts (`scripts/convert_legacy_model.py`, `scripts/convert_legacy_null.py`)
with an explicit `--trusted-input` security guard. The Julia package never
reads `pickle` or `joblib` files.