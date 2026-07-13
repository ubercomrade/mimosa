# ADR 0007: CLI and Distribution

## Status

Accepted and implemented. Amended for the profile-only public contract.

## Decision

Use a dependency-free custom parser in `Mimosa.jl/src/cli.jl` with the
standalone entry point `Mimosa.jl/app/mimosa.jl`. The CLI remains a thin adapter
over public library workflows and contains no independent scientific logic.

Current commands are:

- `profile`;
- `build-null`;
- `cache clear`;
- `inspect-model`;
- `convert-model`.

The direct `motif` command and a `convert-null` command are not available.

## Process Contract

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | usage or argument error |
| 2 | runtime, input, or scientific error |

Successful machine-readable results are written only to stdout as JSON.
Diagnostics and errors are written only to stderr. Batch mode has no prompts,
and stack traces are hidden unless verbose diagnostics are requested.

`profile` accepts model or score-profile inputs and supports only `co`,
`co_rowwise`, `dice`, `dice_rowwise`, and `cosine`. `build-null` always builds a
profile null distribution. `--threads` selects an explicit execution policy but
does not alter Julia's runtime thread count.

## Consequences

- CLI behavior is covered through direct and subprocess tests.
- JSON schemas, exit codes, and stdout/stderr separation are public contracts.
- Legacy unsafe serialization is never read by the CLI; `convert-model`
  converts supported scientific model files into portable Mimosa bundles.
