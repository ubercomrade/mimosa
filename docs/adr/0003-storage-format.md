# ADR 0003: Storage Format

## Status

Proposed (Stage 0)

## Context

The Python implementation persists models and null distributions using `joblib.dump`/`joblib.load`, which is Python pickle under the hood. This is:
- **Unsafe**: pickle can execute arbitrary code on load.
- **Non-portable**: tied to Python, NumPy, and the exact `GenericModel` class definition.
- **Fragile**: format changes break old files silently.
- **Opaque**: no schema, no version, no checksums.

`Mimosa.jl` needs a language-neutral, versioned, safe, and portable storage format for models and null distributions.

## Decision

Adopt a **JSON manifest + NPY-compatible binary blobs** bundle format. The bundle is either a directory (for local use) or a ZIP container (for single-file distribution).

### Directory bundle structure

```
my_model.mimosa/
├── manifest.json
├── data/
│   ├── weights.npy        # or log_odds.npy, frequencies.npy
│   └── background.npy     # optional
```

### ZIP bundle structure

```
my_model.mimosa            # ZIP file with .mimosa extension
├── manifest.json
├── data/weights.npy
└── data/background.npy
```

### Manifest schema (v1)

```json
{
  "format": "mimosa",
  "format_version": 1,
  "kind": "model",
  "model_type": "pwm",
  "name": "pwm_model",
  "motif_length": 12,
  "model_specific": {
    "background": [0.25, 0.25, 0.25, 0.25]
  },
  "arrays": {
    "weights": {
      "file": "data/weights.npy",
      "dtype": "<f4",
      "shape": [4, 12],
      "layout": "base_position",
      "checksum": "sha256:..."
    }
  },
  "provenance": {
    "source_format": "meme",
    "source_path": null,
    "tool_version": null,
    "created_at": null
  },
  "coordinate_convention": {
    "indexing": "one_based",
    "orientation": "forward"
  }
}
```

### Null distribution manifest (v1)

```json
{
  "format": "mimosa",
  "format_version": 1,
  "kind": "null_distribution",
  "strategy": "motif",
  "metric": "pcc",
  "estimator_type": "genextreme",
  "genextreme_params": [shape, location, scale],
  "n_null": 5000,
  "compatibility": {
    "format_version": 1,
    "strategy": "motif",
    "metric": "pcc",
    "sequence_fingerprint": "...",
    "background_fingerprint": "...",
    "model_collection_fingerprint": "...",
    "relation_fingerprint": "..."
  },
  "arrays": {
    "raw_null_scores": {
      "file": "data/raw_null_scores.npy",
      "dtype": "<f8",
      "shape": [5000],
      "checksum": "sha256:..."
    }
  }
}
```

### Safety requirements

1. **Size limits**: declared array shapes must not exceed configurable maxima (default: 100M elements per array, 1GB total bundle size).
2. **Checksum validation**: every binary blob has a SHA-256 checksum in the manifest. Load fails on mismatch.
3. **Path traversal**: ZIP entries must not contain `..` or absolute paths. Reject on extraction.
4. **No code execution**: JSON manifest is parsed with a strict parser; no `eval`, no `deserialize`.
5. **NPY format**: binary blobs use the standard NumPy `.npy` format (magic, version, header, data). This is language-neutral and well-documented. Julia reads it without NumPy.

### Version policy

- `format_version` is a monotonically increasing integer.
- Readers must support all versions ≤ current.
- Writers always write the current version.
- Schema-breaking changes increment the version and require a migration path.

## Alternatives considered

### A. HDF5 (JLD2/HDF5.jl)

Considered: mature, supports partial reads, binary.
Rejected as primary format because:
- HDF5 dependency is heavy (compiles, install time).
- HDF5's own versioning and feature set is complex.
- Security: HDF5 has had CVEs; the attack surface is larger than JSON+NPY.
- Python interop requires `h5py`, another dependency.

Could be added as an optional extension for users who need HDF5 interop, but not the primary format.

### B. MessagePack

Considered: more compact than JSON, binary, language-neutral.
Rejected because:
- JSON is human-readable for the manifest, which aids debugging.
- MessagePack requires a Julia dependency; JSON stdlib is free.
- The bulk of data is in binary blobs (NPY), not the manifest. Manifest size is negligible.

### C. Julia Serialization

Explicitly rejected:
- Tied to Julia version and type definitions.
- Unsafe (can instantiate arbitrary types).
- Not portable to Python or other languages.
- Allowed only for temporary local cache (e.g., precompiled scratch), never as user-facing format.

### D. Arrow

Considered for null distribution scores (tabular data).
Rejected as primary format because:
- Model data is n-dimensional, not tabular.
- Arrow.jl dependency for a non-tabular use case is wasteful.
- NPY is simpler for the actual data shapes.

## Consequences

- Model files written by Julia are readable by a Python script (JSON + NPY are both Python-readable).
- Legacy `joblib`/`pickle` files are NOT readable by Julia core. A separate Python converter script (`scripts/convert_legacy_model.py`) with `--trusted-input` flag handles migration.
- Null distribution files are self-describing: manifest contains compatibility metadata for matching against comparison configs.
- ZIP container support is deferred to Stage 7; directory bundles suffice for Stage 1.
- NPY writer is simple (magic + version + header + data); no external Julia dependency needed.

## Migration impact

- Python `joblib.dump(model, path)` → Julia `writemodel(path, model)`.
- Python `joblib.load(path)` → Julia `readmodel(path)`.
- Existing `.joblib` files require conversion via `scripts/convert_legacy_model.py --trusted-input`.
- Null distribution `.joblib` files require conversion via `scripts/convert_legacy_null.py --trusted-input`.
- The converter is a Python script (not Julia) because it needs to load pickle.