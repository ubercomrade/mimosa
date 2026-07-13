# ADR 0008: Portable Bundle Hardening

## Status

Accepted for the release-candidate remediation slice (2026-07-12).

## Context

Model and null bundles are untrusted TOML plus NPY input. The original readers
validated only a subset of the manifest and read the remainder of an NPY file
as payload. That allowed path traversal, checksum bypasses, allocation denial
and accidental propagation of parser exceptions such as `KeyError` or
`BoundsError`.

## Decision

Use one shared bundle boundary for model and null storage:

- manifest versions must be positive integers matching the supported schema:
  model v1 or null v2;
- referenced files must use relative forward-slash paths with no `.`/`..`,
  drive prefixes or backslashes, and their resolved targets must remain under
  the bundle root;
- every array requires `sha256:` followed by exactly 64 lowercase hex digits;
- manifests, arrays, dimensions, ranks, element counts and total declared
  payload are bounded before NPY output allocation;
- NPY readers validate magic, version, little-endian float dtype, header
  structure, rank, shape, row-major order and exact payload length;
- model-specific shape and order/span invariants are checked from the manifest
  before reading the blob;
- malformed bundles are reported as `ModelFormatError`; writer-side
  consistency failures use `InvariantError`;
- writers build a complete sibling staging directory and commit it with one
  directory rename. Orphan `.mimosa-stage-*` directories are never adopted by
  readers; the current operation removes its stage best-effort, while an
  orphan left after process termination is safe to inspect or remove manually.

The current limits are 1 MiB per manifest, 1 GiB per blob, 64 arrays, rank 8,
100,000,000 elements per array/dimension, and 1 GiB total declared allocation.
These are on-disk trust-boundary limits and do not change ordinary in-memory
model constructors.

## Consequences

The existing model-v1 and null-v2 TOML schemas and canonical `data/*.npy` names
remain compatible.
The reader intentionally supports only the little-endian Float32/Float64 NPY
types used by the v1 writers. A future schema or dtype extension requires a
new reviewed format version rather than permissive fallback parsing.
