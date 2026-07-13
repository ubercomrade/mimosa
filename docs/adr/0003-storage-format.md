# ADR 0003: Storage Format

## Status

Accepted and implemented. Model format version 1 and null format version 2 are
the current contracts.

## Context

User-facing model and null storage must be portable, inspectable, bounded, and
safe for untrusted input. Python pickle/joblib and Julia `Serialization` can
execute or instantiate code and are not stable cross-language formats.

## Decision

Use directory bundles containing a bounded TOML manifest and one or more
checksum-verified NPY blobs:

```text
bundle/
├── manifest.toml
└── data/
    └── values.npy
```

Model bundles use format version 1 and support `pwm`, `pfm`, `bamm`, `sitega`,
`dimont`, and `slim`. Null bundles use format version 2, store Float64 raw null
scores and GEV metadata, and require strategy `"profile"` with one of the
supported profile metric names.

In-memory matrices follow Julia column-major conventions. NPY blobs are written
in explicit row-major order for interoperability. Every blob declaration
contains a relative path, dtype, shape, and exact lowercase SHA-256 checksum.

Readers reject path traversal and symlink escape, malformed or unsupported NPY
headers, unsupported dtypes/endianness, shape or payload mismatch, non-finite
scientific values, unsupported schema versions, and oversized declarations
before allocation.

Writers assemble a complete sibling staging directory and commit it with a
single rename. A failed or interrupted write cannot replace a valid target;
orphan staging directories are ignored by readers.

## Version Policy

Writers emit only the current format. Readers accept only explicitly supported
versions and reject newer or obsolete incompatible versions. A deliberate
schema change requires a new constant, migration/compatibility checks, tests,
and updated documentation.

## Consequences

- `writemodel`/`readmodel` and `savenull`/`loadnull` are the supported portable
  storage APIs.
- ZIP containers are not part of the current format.
- Legacy pickle/joblib data must be converted outside the Julia trust boundary;
  the package never deserializes it directly.
- Cache storage follows the same bounded/atomic principles with its own format
  version 1.
